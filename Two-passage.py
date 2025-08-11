#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# HeatSafe Whisper Transcriber — ULTRA-профиль качества без перегрева.
# Новое:
# - beam patience (лучше поиск при beam>1)
# - "context tail" → initial_prompt на каждый чанк (связность)
# - 2-й проход только по "сомнительным" коротким сегментам (beam↑, patience↑)
# - VRAM-guard: мягкие паузы, когда VRAM близка к потолку
# Остальное: денойз, троттлинг по температуре, чанкование, прогресс/телеметрия.

import argparse
import os, re, sys, time, signal, subprocess
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

from rich import print
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.align import Align
from rich.progress import (
  Progress, BarColumn, TextColumn, TimeElapsedColumn,
  TimeRemainingColumn, TaskProgressColumn, SpinnerColumn
)

from faster_whisper import WhisperModel

# ------------------ media/system utils ------------------

def call_ffprobe_duration(path: str) -> float:
  try:
    out = subprocess.check_output(
      ["ffprobe","-v","error","-show_entries","format=duration",
       "-of","default=noprint_wrappers=1:nokey=1", path],
      stderr=subprocess.STDOUT
    ).decode().strip()
    return float(out)
  except Exception:
    return 0.0

def ffmpeg_denoise(src: str, dst: str) -> None:
  cmd = [
    "ffmpeg","-hide_banner","-y","-loglevel","error",
    "-i", src, "-vn",
    "-ac","1","-ar","16000",
    "-af","highpass=f=80,lowpass=f=8000,afftdn=nr=24,dynaudnorm",
    "-c:a","pcm_s16le", dst
  ]
  try:
    subprocess.check_call(cmd)
  except subprocess.CalledProcessError:
    subprocess.check_call([
      "ffmpeg","-hide_banner","-y","-loglevel","error",
      "-i",src,"-vn","-ac","1","-ar","16000","-c:a","pcm_s16le",dst
    ])

def has_cuda() -> bool:
  try:
    import ctranslate2
    return getattr(ctranslate2, "get_cuda_device_count", lambda: 0)() > 0
  except Exception:
    return False

def nvidia_smi_query(fields: List[str]) -> List[str]:
  try:
    out = subprocess.check_output(
      ["nvidia-smi","--query-gpu="+",".join(fields),"--format=csv,noheader,nounits"],
      stderr=subprocess.STDOUT
    ).decode().strip().splitlines()
    return out
  except Exception:
    return []

def get_gpu_stats() -> Dict[str,float]:
  lines = nvidia_smi_query(["temperature.gpu","utilization.gpu","memory.used","memory.total"])
  if not lines:
    return {"temp":-1.0,"util":-1.0,"mem_used":-1.0,"mem_total":-1.0}
  vals = [v.strip() for v in lines[0].split(",")]
  try:
    return {"temp":float(vals[0]),"util":float(vals[1]),"mem_used":float(vals[2]),"mem_total":float(vals[3])}
  except Exception:
    return {"temp":-1.0,"util":-1.0,"mem_used":-1.0,"mem_total":-1.0}

def get_gpu_compute_procs(limit:int=3)->List[Tuple[str,str,str]]:
  try:
    out = subprocess.check_output(
      ["nvidia-smi","--query-compute-apps=pid,process_name,used_gpu_memory","--format=csv,noheader,nounits"],
      stderr=subprocess.STDOUT
    ).decode().strip().splitlines()
    rows=[]
    for line in out[:limit]:
      pid,name,mem=[x.strip() for x in line.split(",")]
      rows.append((pid, name, mem+" MiB"))
    return rows
  except Exception:
    return []

def get_cpu_temp_max() -> float:
  try:
    temps=[]
    for root,_,files in os.walk("/sys/class/thermal"):
      for fn in files:
        if fn=="temp":
          p=os.path.join(root,fn)
          try:
            v=int(open(p).read().strip())
            temps.append(v/1000.0 if v>1000 else float(v))
          except Exception:
            pass
    return max(temps) if temps else -1.0
  except Exception:
    return -1.0

def get_top_cpu_procs(limit:int=3)->List[Tuple[str,str,str,str]]:
  try:
    out = subprocess.check_output(
      ["bash","-lc",f"ps -eo pid,comm,%cpu,%mem --sort=-%cpu | sed -n '2,{limit+1}p'"],
      stderr=subprocess.STDOUT
    ).decode().strip().splitlines()
    rows=[]
    for line in out:
      parts=line.split(None,3)
      if len(parts)==4: rows.append(tuple(parts))
    return rows
  except Exception:
    return []

# ------------------ chunking ------------------

def ensure_chunks(path: str, chunk_seconds: int, workdir: str) -> List[str]:
  os.makedirs(workdir, exist_ok=True)
  existing = sorted([os.path.join(workdir,f) for f in os.listdir(workdir) if f.startswith("chunk_")])
  if existing: return existing
  ext = os.path.splitext(path)[1].lower() or ".m4a"
  outpat = os.path.join(workdir, f"chunk_%04d{ext}")
  subprocess.check_call([
    "ffmpeg","-hide_banner","-loglevel","error",
    "-i", path, "-c","copy",
    "-f","segment","-segment_time", str(chunk_seconds),
    "-reset_timestamps","1", outpat
  ])
  return sorted([os.path.join(workdir,f) for f in os.listdir(workdir) if f.startswith("chunk_")])

# ------------------ model policy ------------------

def device_order(pref:str)->List[str]:
  if pref=="cuda": return ["cuda"]
  if pref=="cpu": return ["cpu"]
  return ["cuda","cpu"] if has_cuda() else ["cpu"]

def compute_candidates(dev:str)->List[str]:
  if dev=="cuda": return ["float16","int8_float16","int8","float32"]
  return ["int8_float16","int8","float32"]

@dataclass
class ModelSpec:
  name:str; device:str; compute:str

def try_load(model_name:str, devs:List[str], forced_compute:Optional[str])->Optional[ModelSpec]:
  for d in devs:
    cands=[forced_compute] if forced_compute else compute_candidates(d)
    for ct in cands:
      try:
        print(f"[cyan]loading model[/cyan] [bold]{model_name}[/bold] on [bold]{d}[/bold] ({ct}) ...")
        _=WhisperModel(model_name, device=d, compute_type=ct)
        return ModelSpec(model_name,d,ct)
      except Exception as e:
        print(f"[yellow]failed[/yellow] on {d} ({ct}): {e}")
  return None

def build_model(spec:ModelSpec)->WhisperModel:
  return WhisperModel(spec.name, device=spec.device, compute_type=spec.compute)

# ------------------ UI ------------------

def render_dashboard(total_sec: float, done_sec: float, chunk_idx:int, chunks_total:int,
                     spec:ModelSpec, beam:int, vad:bool,
                     max_gpu:float, max_cpu:float,
                     max_vram_frac:float, patience:float)->Panel:
  gpu = get_gpu_stats(); cpu_t = get_cpu_temp_max()
  usage = f"{done_sec/total_sec*100:.1f}%" if total_sec>0 else "-"
  t = Table.grid(padding=(0,2))
  sys_table = Table.grid()
  sys_table.add_row(
    f"[bold]GPU[/bold] {gpu['temp']:.0f}°C" if gpu['temp']>=0 else "[bold]GPU[/bold] n/a",
    f"util {gpu['util']:.0f}%" if gpu['util']>=0 else "",
    f"VRAM {gpu['mem_used']:.0f}/{gpu['mem_total']:.0f} MiB" if gpu['mem_total']>0 else ""
  )
  sys_table.add_row(
    f"[bold]CPU[/bold] {cpu_t:.0f}°C" if cpu_t>=0 else "[bold]CPU[/bold] n/a",
    f"[dim]limits[/dim] GPU≤{int(max_gpu)}° CPU≤{int(max_cpu)}° VRAM≤{int(max_vram_frac*100)}%"
  )
  mdl_table = Table.grid()
  mdl_table.add_row(f"[bold]Model[/bold] {spec.name}")
  mdl_table.add_row(f"[bold]Device[/bold] {spec.device}  [bold]Compute[/bold] {spec.compute}")
  mdl_table.add_row(f"[bold]Beam[/bold] {beam}  [bold]Patience[/bold] {patience:g}  [bold]VAD[/bold] {str(vad)}")
  mdl_table.add_row(f"[bold]Chunk[/bold] {chunk_idx+1}/{chunks_total}")
  proc_table = Table.grid()
  cpu_top = get_top_cpu_procs()
  if cpu_top:
    proc_table.add_row("[bold]Top CPU[/bold]")
    for pid,comm,cpu,mem in cpu_top:
      proc_table.add_row(f"{comm}({pid})  {cpu}% cpu  {mem}% mem")
  gtop = get_gpu_compute_procs()
  if gtop:
    proc_table.add_row(""); proc_table.add_row("[bold]GPU procs[/bold]")
    for pid,name,mem in gtop:
      proc_table.add_row(f"{os.path.basename(name)}({pid})  {mem}")
  t.add_row(sys_table, mdl_table, proc_table)
  title = f"HeatSafe Transcriber • progress {usage}" if total_sec>0 else "HeatSafe Transcriber"
  return Panel(t, title=title, padding=(1,2))

# ------------------ guards ------------------

def maybe_cooldown(max_gpu:float, max_cpu:float, cooldown_base:float)->bool:
  gpu = get_gpu_stats(); cpu_t = get_cpu_temp_max()
  hot_gpu = gpu["temp"]>=0 and max_gpu>0 and gpu["temp"]>=max_gpu
  hot_cpu = cpu_t>=0 and max_cpu>0 and cpu_t>=max_cpu
  if hot_gpu or hot_cpu:
    over=0.0
    if hot_gpu: over=max(over, gpu["temp"]-max_gpu)
    if hot_cpu: over=max(over, cpu_t-max_cpu)
    sleep_s = cooldown_base * (1.0 + min(1.0, over/8.0))
    print(f"[yellow]thermal guard:[/yellow] cooldown {sleep_s:.1f}s (gpu={gpu['temp']:.0f}°, cpu={cpu_t:.0f}°)")
    time.sleep(sleep_s); return True
  return False

def maybe_vram_guard(max_frac:float, cooldown_base:float)->bool:
  g=get_gpu_stats()
  if g["mem_total"]>0 and g["mem_used"]>=0:
    frac = g["mem_used"]/g["mem_total"]
    if frac >= max_frac:
      sleep_s = cooldown_base * (0.8 + (frac-max_frac)*4)  # чуть адаптивнее
      print(f"[yellow]vram guard:[/yellow] usage={frac*100:.0f}% ≥ {max_frac*100:.0f}% → pause {sleep_s:.1f}s")
      time.sleep(sleep_s); return True
  return False

# ------------------ text helpers ------------------

FILLERS = [r"\bну\b", r"\bкак бы\b", r"\bтипа\b", r"\bв общем\b", r"\bкороче\b",
           r"\bэто самое\b", r"\bпросто\b", r"\bкак-то\b", r"\bвот\b"]

def cleanup_fillers(text:str)->str:
  x=text
  for pat in FILLERS:
    x=re.sub(rf"(?:{pat})(\s+\1)+", r"\1", x, flags=re.IGNORECASE)
  return x

def soft_wrap(text:str, width:int)->str:
  if width<=0: return text.strip()
  words=text.strip().split()
  if not words: return ""
  lines,cur=[],words[0]
  for w in words[1:]:
    if len(cur)+1+len(w)<=width: cur+=" "+w
    else: lines.append(cur); cur=w
  lines.append(cur); return "\n".join(lines)

# ------------------ main ------------------

def main():
  ap=argparse.ArgumentParser(description="Local high-quality transcription (ULTRA), with chunking, denoise, telemetry, and safety guards.")
  ap.add_argument("input")
  ap.add_argument("--language", default=None)
  ap.add_argument("--device", default="auto", choices=["auto","cuda","cpu"])
  ap.add_argument("--model", default="large-v3")
  ap.add_argument("--fallback-models", default="large-v3,medium,small")
  ap.add_argument("--compute", default=None, choices=["int8","int8_float16","float16","float32"])
  ap.add_argument("--beam", type=int, default=5)
  ap.add_argument("--patience", type=float, default=0.0, help="beam patience (0=off). ULTRA рекомендует 1.0")
  ap.add_argument("--vad-off", action="store_true")
  ap.add_argument("--max-chars", type=int, default=0)
  ap.add_argument("--out", default=None)
  ap.add_argument("--mode", default="balanced", choices=["fast","balanced","cool","ultra"])
  ap.add_argument("--denoise", action="store_true")
  ap.add_argument("--word-ts", action="store_true")
  ap.add_argument("--bias", default=None, help="comma-separated domain words/phrases")
  ap.add_argument("--clean-fillers", action="store_true")
  ap.add_argument("--chunk-seconds", type=int, default=600)
  ap.add_argument("--workdir", default=None)
  ap.add_argument("--max-temp-gpu", type=float, default=78.0)
  ap.add_argument("--max-temp-cpu", type=float, default=85.0)
  ap.add_argument("--cooldown-seconds", type=float, default=8.0)
  # контекст и второй проход
  ap.add_argument("--context-tail-chars", type=int, default=280, help="символов из конца предыдущего контекста в initial_prompt")
  ap.add_argument("--second-pass", action="store_true", help="перепрожиг сомнительных коротких сегментов")
  ap.add_argument("--sp-threshold", type=float, default=-0.9, help="avg_logprob ниже — кандидат на 2-й проход")
  ap.add_argument("--sp-max-secs", type=float, default=25.0, help="макс. длительность сегмента для 2-го прохода")
  ap.add_argument("--sp-beam", type=int, default=8, help="beam для 2-го прохода")
  ap.add_argument("--sp-patience", type=float, default=1.5, help="patience для 2-го прохода")
  # VRAM guard
  ap.add_argument("--max-vram-frac", type=float, default=0.92, help="порог доли VRAM для паузы")
  args=ap.parse_args()

  in_path=args.input
  if not os.path.exists(in_path):
    print(f"[red]file not found: {in_path}[/red]"); sys.exit(1)

  # presets
  if args.mode=="fast":
    args.beam=max(3,args.beam); args.patience=max(args.patience,0.0)
    if args.compute is None: args.compute="float16" if has_cuda() else "float32"
    if args.chunk_seconds==0: args.chunk_seconds=600
  elif args.mode=="cool":
    args.beam=min(args.beam,2); args.patience=min(args.patience,0.5)
    if args.compute is None: args.compute="int8_float16" if has_cuda() else "int8"
    if args.chunk_seconds==0 or args.chunk_seconds>480: args.chunk_seconds=480
  elif args.mode=="ultra":
    args.beam=max(args.beam,7); args.patience=max(args.patience,1.0)
    if args.compute is None: args.compute="int8_float16" if has_cuda() else "float32"
    if args.chunk_seconds==0 or args.chunk_seconds>480: args.chunk_seconds=480
    if not args.denoise: args.denoise=True
  else:
    args.beam=min(args.beam,3)
    if args.patience==0.0: args.patience=0.5
    if args.compute is None: args.compute="float16" if has_cuda() else "int8_float16"
    if args.chunk_seconds==0: args.chunk_seconds=600

  total_sec=call_ffprobe_duration(in_path)
  if total_sec<=0: print("[yellow]ffprobe: unknown duration — ETA может быть неточен[/yellow]")

  base = args.out if args.out else os.path.splitext(in_path)[0]
  workdir = args.workdir if args.workdir else base+"_chunks"

  # chunks
  chunks = ensure_chunks(in_path, args.chunk_seconds, workdir) if args.chunk_seconds>0 else [in_path]
  chunks_total=len(chunks)

  # model select
  devs=device_order(args.device)
  wanted=[m.strip() for m in args.fallback_models.split(",") if m.strip()]
  gpu=get_gpu_stats()
  if gpu["mem_total"]>0 and gpu["mem_total"]<=4096 and "large-v3" in wanted:
    print("[yellow]VRAM≈4GiB → starting from medium[/yellow]")
    wanted=[m for m in wanted if m!="large-v3"]
    if "medium" not in wanted: wanted.insert(0,"medium")
  spec=None
  for m in wanted:
    spec=try_load(m,devs,args.compute)
    if spec: break
  if not spec:
    print("[red]could not initialize any model[/red]"); sys.exit(2)
  model=build_model(spec)
  vad = not args.vad_off

  # SIGINT graceful
  stop_flag={"stop":False}
  def on_sigint(sig,frame):
    stop_flag["stop"]=True
    print("\n[yellow]interrupt — завершаю текущий чанк и сохраняю результаты[/yellow]")
  signal.signal(signal.SIGINT, on_sigint)

  # progress
  total_done=0.0
  prog_main=Progress(SpinnerColumn(), TextColumn("[bold blue]total[/bold blue]"),
                     BarColumn(), TaskProgressColumn(), TextColumn("•"),
                     TimeElapsedColumn(), TextColumn("• ETA:"), TimeRemainingColumn(), transient=False)
  prog_chunk=Progress(SpinnerColumn(), TextColumn("[bold magenta]chunk[/bold magenta]"),
                      BarColumn(), TaskProgressColumn(), TextColumn("•"),
                      TimeElapsedColumn(), TextColumn("• ETA:"), TimeRemainingColumn(), transient=False)
  task_total=prog_main.add_task("total", total=total_sec if total_sec>0 else None, completed=0)
  task_chunk=None

  segments: List[Dict[str,Any]]=[]
  chunk_offsets: List[float]=[]
  context_tail=""

  def render_group(idx:int)->Panel:
    return render_dashboard(total_sec, total_done, idx, chunks_total, spec, args.beam, vad,
                            args.max_temp_gpu, args.max_temp_cpu, args.max_vram_frac, args.patience)

  # helper: main transcribe call with our quality knobs
  def run_transcribe(input_path: str, beam:int, patience:float, init_prompt:str):
    return model.transcribe(
      input_path,
      language=args.language,
      beam_size=beam,
      patience=patience if beam>1 else 0.0,
      vad_filter=vad,
      vad_parameters=dict(min_silence_duration_ms=500),
      word_timestamps=args.word_ts,
      condition_on_previous_text=True,
      temperature=[0.0, 0.2, 0.4],
      compression_ratio_threshold=2.3,
      no_speech_threshold=0.60,
      initial_prompt=(init_prompt if init_prompt else None),
    )

  # -------- main loop --------
  with Live(Panel.fit("initializing...", border_style="gray50"), refresh_per_second=2) as live:
    live.update(Panel.fit("preparing...", border_style="gray50"))
    while chunks:
      if stop_flag["stop"]: break
      cur_idx=len(chunk_offsets)
      chunk_path=chunks.pop(0)
      ch_dur=call_ffprobe_duration(chunk_path)

      # clean
      clean_chunk=chunk_path
      if args.denoise:
        clean_chunk=os.path.join(workdir, f"clean_{os.path.basename(chunk_path)}.wav")
        ffmpeg_denoise(chunk_path, clean_chunk)

      if task_chunk is not None: prog_chunk.remove_task(task_chunk)
      task_chunk = prog_chunk.add_task("chunk", total=ch_dur if ch_dur>0 else None, completed=0)

      group=Table.grid(padding=1); group.add_row(render_group(cur_idx)); group.add_row(prog_main); group.add_row(prog_chunk)
      live.update(group)

      # safety precheck
      maybe_vram_guard(args.max_vram_frac, args.cooldown_seconds)
      if maybe_cooldown(args.max_temp_gpu, args.max_temp_cpu, args.cooldown_seconds):
        group=Table.grid(padding=1); group.add_row(render_group(cur_idx)); group.add_row(prog_main); group.add_row(prog_chunk)
        live.update(group)

      offset=sum(chunk_offsets)

      # build contextual prompt: bias + tail
      bias = (args.bias or "").strip()
      init_prompt = ""
      if bias and context_tail:
        init_prompt = f"{bias}. {context_tail}"
      elif bias:
        init_prompt = bias
      else:
        init_prompt = context_tail

      try:
        seg_iter, info = run_transcribe(clean_chunk, args.beam, args.patience, init_prompt)
        last_local=0.0
        # локальный список сегментов этого чанка для 2-го прохода
        indices_this_chunk: List[int]=[]
        for s in seg_iter:
          if stop_flag["stop"]: break

          # guards mid-run
          if maybe_vram_guard(args.max_vram_frac, args.cooldown_seconds) or \
             maybe_cooldown(args.max_temp_gpu, args.max_temp_cpu, args.cooldown_seconds):
            # мягче: сначала beam -> 1; потом compute → экономичнее
            if args.beam>1:
              args.beam=1
            else:
              next_ct=None
              if spec.device=="cuda":
                for ct in ["int8_float16","int8","float32"]:
                  if ct!=spec.compute: next_ct=ct; break
              if next_ct:
                print(f"[yellow]switch compute → {next_ct} (cooldown)[/yellow]")
                spec=ModelSpec(spec.name, spec.device, next_ct); model=build_model(spec)
            seg_iter, info = run_transcribe(clean_chunk, args.beam, args.patience, init_prompt)
            last_local=0.0
            prog_chunk.reset(task_chunk, total=ch_dur if ch_dur>0 else None, completed=0)

          text=(s.text or "").strip()
          if args.max_chars>0 and len(text)>args.max_chars:
            text=soft_wrap(text, args.max_chars)

          # сохраняем доп. метрики для 2-го прохода
          seg_dict={
            "start": offset + (s.start or 0.0),
            "end":   offset + (s.end or 0.0),
            "text":  text,
            "avg_logprob": getattr(s,"avg_logprob", None),
            "no_speech_prob": getattr(s,"no_speech_prob", None),
            "chunk_idx": cur_idx,
            "local_start": float(s.start or 0.0),
            "local_end": float(s.end or 0.0)
          }
          segments.append(seg_dict)
          indices_this_chunk.append(len(segments)-1)

          # progress
          local=float(s.end or 0.0)
          if ch_dur>0 and local>last_local:
            prog_chunk.update(task_chunk, completed=min(local, ch_dur)); last_local=local
          if total_sec>0:
            total_done=min(total_sec, offset+local); prog_main.update(task_total, completed=total_done)

          if int(time.time())%2==0:
            group=Table.grid(padding=1); group.add_row(render_group(cur_idx)); group.add_row(prog_main); group.add_row(prog_chunk)
            live.update(group)

        if stop_flag["stop"]: break

        # ---------- Второй проход (точечный) ----------
        if args.second_pass and indices_this_chunk:
          improved=0
          for idx in indices_this_chunk:
            seg=segments[idx]
            dur = seg["local_end"] - seg["local_start"]
            avg_lp = seg.get("avg_logprob", None)
            bad_lp = (avg_lp is not None and avg_lp <= args.sp_threshold)
            tiny = (len(seg["text"])<=3)
            if dur>0 and dur<=args.sp_max_secs and (bad_lp or tiny):
              # вырезаем точный кусок из clean_chunk
              trim_path = os.path.join(workdir, f"sp_{cur_idx}_{idx}.wav")
              ss = max(0.0, seg["local_start"]); to = max(ss, seg["local_end"])
              subprocess.check_call([
                "ffmpeg","-hide_banner","-y","-loglevel","error",
                "-ss", f"{ss:.3f}","-to", f"{to:.3f}",
                "-i", clean_chunk, "-ac","1","-ar","16000","-c:a","pcm_s16le", trim_path
              ])
              # контекст для подчищаемого куска: хвост до него
              around_tail = ""
              # возьмем до 2-х предыдущих сегментов этого чанка
              prev_texts=[]
              for j in range(max(indices_this_chunk[0], idx-2), idx):
                prev_texts.append(segments[j]["text"])
              if prev_texts:
                joined=" ".join(prev_texts)[-args.context_tail_chars:]
                around_tail = (args.bias + ". " if args.bias else "") + joined

              sp_iter, _ = run_transcribe(trim_path, max(args.sp_beam, args.beam), max(args.sp_patience, args.patience), around_tail)
              sp_text_parts=[]
              for t in sp_iter:
                part=(t.text or "").strip()
                if part: sp_text_parts.append(part)
              if sp_text_parts:
                new_text=" ".join(sp_text_parts)
                if args.max_chars>0 and len(new_text)>args.max_chars:
                  new_text=soft_wrap(new_text, args.max_chars)
                segments[idx]["text"]=new_text
                improved+=1
          if improved>0:
            print(f"[green]second pass:[/green] refined {improved} segment(s) in chunk {cur_idx+1}")

      except RuntimeError as e:
        emsg=str(e).lower()
        if "out of memory" in emsg or "cuda error" in emsg:
          print("[yellow]OOM/driver issue — fallback to smaller model[/yellow]")
          order=[m.strip() for m in args.fallback_models.split(",") if m.strip()]
          try_idx = order.index(spec.name) if spec.name in order else -1
          fallback_spec=None
          for nm in order[try_idx+1:]:
            cand=try_load(nm, device_order(args.device), args.compute)
            if cand: fallback_spec=cand; break
          if not fallback_spec:
            print("[red]no smaller model available[/red]"); sys.exit(4)
          spec=fallback_spec; model=build_model(spec)
          chunks.insert(0, chunk_path); continue
        else:
          raise

      # chunk finished
      chunk_offsets.append(ch_dur)
      if total_sec>0:
        total_done=min(total_sec, sum(chunk_offsets)); prog_main.update(task_total, completed=total_done)

      # обновим контекстный хвост для следующего чанка
      if segments:
        # берём последние 2-3 реплики текущего чанка
        last_texts=[]
        for j in range(len(segments)-1, -1, -1):
          if segments[j]["chunk_idx"]!=cur_idx: break
          last_texts.append(segments[j]["text"])
          if len(last_texts)>=3: break
        context_tail = " ".join(reversed(last_texts))[-args.context_tail_chars:]

      group=Table.grid(padding=1); group.add_row(render_group(cur_idx)); group.add_row(prog_main); group.add_row(prog_chunk)
      live.update(group)

  # пост-обработка
  if args.clean_fillers:
    for seg in segments:
      seg["text"]=cleanup_fillers(seg["text"])

  # запись
  def ts_srt(t:float)->str:
    t=max(0.0,t); hh=int(t//3600); mm=int((t%3600)//60); ss=int(t%60); ms=int(round((t-int(t))*1000))
    return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"
  def ts_vtt(t:float)->str:
    return ts_srt(t).replace(",", ".")

  txt_path=base+".txt"; srt_path=base+".srt"; vtt_path=base+".vtt"
  with open(txt_path,"w",encoding="utf-8") as f:
    for seg in segments: f.write(seg["text"]+"\n")
  with open(srt_path,"w",encoding="utf-8") as f:
    for i,seg in enumerate(segments,1):
      f.write(f"{i}\n{ts_srt(seg['start'])} --> {ts_srt(seg['end'])}\n{seg['text']}\n\n")
  with open(vtt_path,"w",encoding="utf-8") as f:
    f.write("WEBVTT\n\n")
    for seg in segments:
      f.write(f"{ts_vtt(seg['start'])} --> {ts_vtt(seg['end'])}\n{seg['text']}\n\n")

  print("[green]done[/green]")
  print(f"device: {spec.device}, compute: {spec.compute}, model: {spec.name}")
  print(f"txt: {txt_path}\nsrt: {srt_path}\nvtt: {vtt_path}")

if __name__=="__main__":
  main()
