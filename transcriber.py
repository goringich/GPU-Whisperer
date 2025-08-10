#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# purpose: robust local transcription via faster-whisper with gpu-first logic + thermal throttling
# highlights:
# - auto device: cuda->cpu, or override via --device
# - auto compute type with fallbacks to avoid unsupported precisions
# - auto model fallback: large-v3 -> medium -> small (configurable)
# - outputs: .txt, .srt, .vtt
# - rich progress: processed, %, elapsed, eta
# - safety: thermal guard for cpu/gpu + gentle throttling (sleep + lowering load)
# deps: faster-whisper, rich, ffmpeg (ffprobe), nvidia-smi (for gpu temps)

import argparse
import os
import sys
import time
import subprocess
from typing import Iterable, Dict, Any, List, Tuple

from rich import print
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, TaskProgressColumn
from faster_whisper import WhisperModel

# ---------- small io helpers

def format_timestamp_srt(seconds: float) -> str:
  s = max(0.0, float(seconds))
  hh = int(s // 3600); s -= hh * 3600
  mm = int(s // 60); s -= mm * 60
  ss = int(s); ms = int(round((s - ss) * 1000))
  return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"

def format_timestamp_vtt(seconds: float) -> str:
  return format_timestamp_srt(seconds).replace(",", ".")

def write_txt(path: str, segs: List[Dict[str, Any]]) -> None:
  with open(path, "w", encoding="utf-8") as f:
    for seg in segs:
      f.write(seg["text"].strip() + "\n")

def write_srt(path: str, segs: List[Dict[str, Any]]) -> None:
  with open(path, "w", encoding="utf-8") as f:
    for i, seg in enumerate(segs, 1):
      f.write(f"{i}\n{format_timestamp_srt(seg['start'])} --> {format_timestamp_srt(seg['end'])}\n{seg['text'].strip()}\n\n")

def write_vtt(path: str, segs: List[Dict[str, Any]]) -> None:
  with open(path, "w", encoding="utf-8") as f:
    f.write("WEBVTT\n\n")
    for seg in segs:
      f.write(f"{format_timestamp_vtt(seg['start'])} --> {format_timestamp_vtt(seg['end'])}\n{seg['text'].strip()}\n\n")

def soft_wrap(text: str, width: int) -> str:
  if width <= 0: return text.strip()
  words = text.strip().split()
  if not words: return ""
  lines, cur = [], words[0]
  for w in words[1:]:
    if len(cur) + 1 + len(w) <= width:
      cur += " " + w
    else:
      lines.append(cur); cur = w
  lines.append(cur)
  return "\n".join(lines)

# ---------- media duration via ffprobe

def get_media_duration(path: str) -> float:
  try:
    out = subprocess.check_output(
      ["ffprobe", "-v", "error", "-show_entries", "format=duration",
       "-of", "default=noprint_wrappers=1:nokey=1", path],
      stderr=subprocess.STDOUT
    ).decode().strip()
    return float(out)
  except Exception:
    return 0.0

# ---------- thermal + gpu helpers

def has_cuda() -> bool:
  try:
    import ctranslate2  # closer to faster-whisper runtime
    return getattr(ctranslate2, "get_cuda_device_count", lambda: 0)() > 0
  except Exception:
    return False

def get_gpu_stats() -> Dict[str, float]:
  # returns {'temp': ..., 'util': ..., 'mem_used': ..., 'mem_total': ...}
  try:
    out = subprocess.check_output(
      ["nvidia-smi", "--query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total",
       "--format=csv,noheader,nounits"],
      stderr=subprocess.STDOUT
    ).decode().strip().splitlines()
    # pick first gpu only
    vals = out[0].split(",")
    return {
      "temp": float(vals[0].strip()),
      "util": float(vals[1].strip()),
      "mem_used": float(vals[2].strip()),
      "mem_total": float(vals[3].strip()),
    }
  except Exception:
    return {"temp": -1.0, "util": -1.0, "mem_used": -1.0, "mem_total": -1.0}

def get_cpu_temp_max() -> float:
  # best-effort: read thermal zones, take max; returns -1 if unknown
  try:
    temps = []
    for root, _, files in os.walk("/sys/class/thermal"):
      for fn in files:
        if fn == "temp":
          p = os.path.join(root, fn)
          try:
            v = int(open(p).read().strip())
            # some zones report millidegree C, others degree C; normalize
            temps.append(v/1000.0 if v > 1000 else float(v))
          except Exception:
            pass
    return max(temps) if temps else -1.0
  except Exception:
    return -1.0

def throttle_if_hot(max_gpu: float, max_cpu: float, cooldown: int) -> bool:
  # returns True if throttled (slept)
  gpu = get_gpu_stats()
  cpu_t = get_cpu_temp_max()
  hot_gpu = gpu["temp"] >= 0 and max_gpu > 0 and gpu["temp"] >= max_gpu
  hot_cpu = cpu_t >= 0 and max_cpu > 0 and cpu_t >= max_cpu
  if hot_gpu or hot_cpu:
    reason = []
    if hot_gpu: reason.append(f"gpu {gpu['temp']:.0f}°C")
    if hot_cpu: reason.append(f"cpu {cpu_t:.0f}°C")
    print(f"[yellow]thermal guard:[/yellow] {' & '.join(reason)} ≥ limit, cooldown {cooldown}s")
    time.sleep(max(1, cooldown))
    return True
  return False

# ---------- device/compute/model orchestration

def pick_device(pref: str) -> List[str]:
  if pref == "cuda": return ["cuda"]
  if pref == "cpu": return ["cpu"]
  return (["cuda", "cpu"] if has_cuda() else ["cpu"])

def compute_candidates_for(dev: str) -> List[str]:
  if dev == "cuda":
    return ["float16", "int8_float16", "int8", "float32"]
  return ["int8_float16", "int8", "float32"]

def try_load_model(model_name: str, dev_order: List[str], forced_compute: str | None) -> Tuple[WhisperModel, str, str]:
  last_err = None
  for dev in dev_order:
    cands = [forced_compute] if forced_compute else compute_candidates_for(dev)
    for ct in cands:
      try:
        print(f"[cyan]loading model[/cyan] [bold]{model_name}[/bold] on [bold]{dev}[/bold] ({ct}) ...")
        m = WhisperModel(model_name, device=dev, compute_type=ct)
        return m, dev, ct
      except Exception as e:
        last_err = e
        print(f"[yellow]failed[/yellow] on {dev} ({ct}): {e}")
  raise RuntimeError(f"unable to load model {model_name} on any device/compute; last error: {last_err}")

def transcribe_file(model: WhisperModel, in_path: str, lang: str | None, beam: int, use_vad: bool):
  return model.transcribe(
    in_path,
    language=lang,
    beam_size=beam,
    vad_filter=use_vad,
    vad_parameters=dict(min_silence_duration_ms=500)
  )

# ---------- main

def main():
  p = argparse.ArgumentParser(description="robust local transcription (gpu-first, thermal-safe).")
  p.add_argument("input", help="path to media file (.mp3/.wav/.m4a/.flac/.mp4 etc.)")
  p.add_argument("--model", default="large-v3", help="tiny/base/small/medium/large-v3")
  p.add_argument("--language", default=None, help="force language code, e.g. ru, en")
  p.add_argument("--beam", type=int, default=5, help="beam size")
  p.add_argument("--compute", default=None, choices=["int8", "int8_float16", "float16", "float32"], help="precision override")
  p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="device preference")
  p.add_argument("--max-chars", type=int, default=0, help="wrap subtitle lines to this length (0 disables)")
  p.add_argument("--no-vad", action="store_true", help="disable VAD")
  p.add_argument("--out", default=None, help="output basename without extension")
  p.add_argument("--fallback-models", default="large-v3,medium,small", help="comma list to try if OOM or init fails")
  # safety
  p.add_argument("--max-temp-gpu", type=float, default=80.0, help="max GPU temperature in °C before throttling (0 disables)")
  p.add_argument("--max-temp-cpu", type=float, default=85.0, help="max CPU temperature in °C before throttling (0 disables)")
  p.add_argument("--cooldown-seconds", type=int, default=10, help="sleep seconds when throttling triggers")
  args = p.parse_args()

  in_path = args.input
  if not os.path.exists(in_path):
    print(f"[red]file not found: {in_path}[/red]"); sys.exit(1)

  total_sec = get_media_duration(in_path)
  if total_sec <= 0:
    print("[yellow]warning:[/yellow] ffprobe didn't return duration; progress ETA may be approximate")

  dev_order = pick_device(args.device)
  models_to_try = [m.strip() for m in args.fallback_models.split(",") if m.strip()]

  model = None; used_dev = "cpu"; used_ct = "float32"
  last_exc = None
  for mname in models_to_try:
    try:
      model, used_dev, used_ct = try_load_model(mname, dev_order, args.compute)
      args.model = mname
      break
    except Exception as e:
      last_exc = e
      continue
  if model is None:
    print(f"[red]failed to initialize any model[/red]\n{last_exc}"); sys.exit(2)

  use_vad = not args.no_vad
  print(f"[cyan]transcribing[/cyan] on [bold]{used_dev}[/bold] with [bold]{args.model}[/bold] ({used_ct}), vad={use_vad}, beam={args.beam}")

  try:
    seg_iter, info = transcribe_file(model, in_path, args.language, args.beam, use_vad)
  except RuntimeError as e:
    print(f"[yellow]transcribe failed[/yellow]: {e}")
    idx = models_to_try.index(args.model) if args.model in models_to_try else -1
    seg_iter = None; info = None
    for next_m in models_to_try[idx+1:]:
      try:
        print(f"[cyan]retry with smaller model[/cyan]: {next_m}")
        model, used_dev, used_ct = try_load_model(next_m, dev_order, args.compute)
        args.model = next_m
        seg_iter, info = transcribe_file(model, in_path, args.language, args.beam, use_vad)
        break
      except Exception as e2:
        print(f"[yellow]retry failed[/yellow] with {next_m}: {e2}")
    if seg_iter is None:
      print("[red]all fallbacks failed during transcription[/red]"); sys.exit(3)

  if total_sec <= 0:
    total_sec = float(getattr(info, "duration", 0.0) or 0.0)

  segments: List[Dict[str, Any]] = []
  last_pos = 0.0

  columns = [
    TextColumn("[bold blue]{task.description}"),
    BarColumn(),
    TaskProgressColumn(),
    TextColumn("•"),
    TimeElapsedColumn(),
    TextColumn("• ETA:"),
    TimeRemainingColumn(),
  ]
  with Progress(*columns, transient=False) as progress:
    task = progress.add_task("processing", total=total_sec if total_sec > 0 else None)

    # safe loop with thermal + OOM fallback mid-run
    while True:
      try:
        for s in seg_iter:
          # thermal guard: check every segment
          throttled = throttle_if_hot(args.max_temp_gpu, args.max_temp_cpu, args.cooldown_seconds)
          if throttled:
            # additionally lower load step-by-step
            if args.beam > 1:
              args.beam = 1
              print("[yellow]lowering beam to 1 due to heat[/yellow]")
            elif used_ct in ("float16", "int8_float16") and used_dev == "cuda":
              # try more compact compute if still hot
              try_order = ["int8_float16", "int8", "float32"]
              next_ct = None
              for ct in try_order:
                if ct != used_ct:
                  next_ct = ct; break
              if next_ct:
                print(f"[yellow]switch compute to {next_ct} due to heat[/yellow]")
                model, used_dev, used_ct = try_load_model(args.model, dev_order, next_ct)
                seg_iter, info = transcribe_file(model, in_path, args.language, args.beam, use_vad)
                segments.clear(); last_pos = 0.0
                progress.reset(task, total=total_sec if total_sec > 0 else None, completed=0)
                break  # restart outer while with new iterator
            # if still hot, will keep sleeping on next segments

          text = s.text or ""
          if args.max_chars > 0 and len(text) > args.max_chars:
            text = soft_wrap(text, args.max_chars)
          segments.append(dict(start=s.start, end=s.end, text=text))

          cur = float(s.end or 0.0)
          if total_sec > 0:
            if cur > last_pos:
              progress.update(task, completed=min(cur, total_sec))
              last_pos = cur
        break  # finished normally

      except RuntimeError as e:
        if "out of memory" in str(e).lower():
          print("[yellow]oom on current model; retrying with smaller model[/yellow]")
          idx = models_to_try.index(args.model) if args.model in models_to_try else -1
          fallback = None
          for next_m in models_to_try[idx+1:]:
            try:
              model, used_dev, used_ct = try_load_model(next_m, dev_order, args.compute)
              args.model = next_m
              seg_iter, info = transcribe_file(model, in_path, args.language, args.beam, use_vad)
              segments.clear(); last_pos = 0.0
              progress.reset(task, total=total_sec if total_sec > 0 else None, completed=0)
              fallback = next_m
              break
            except Exception as e2:
              print(f"[yellow]retry failed[/yellow] with {next_m}: {e2}")
          if not fallback:
            print("[red]all fallbacks failed after OOM[/red]"); sys.exit(4)
        else:
          raise

  base = args.out if args.out else os.path.splitext(in_path)[0]
  txt_path = base + ".txt"; srt_path = base + ".srt"; vtt_path = base + ".vtt"

  write_txt(txt_path, segments)
  write_srt(srt_path, segments)
  write_vtt(vtt_path, segments)

  print("[green]done[/green]")
  print(f"device: {used_dev}, compute: {used_ct}, model: {args.model}")
  print(f"txt: {txt_path}\nsrt: {srt_path}\nvtt: {vtt_path}")

if __name__ == "__main__":
  main()
