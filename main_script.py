import os
import subprocess
import re
import torch
import whisperx
from transformers import pipeline, CLIPProcessor, CLIPModel
from PIL import Image

# ==================== CONFIGURATION ====================
SOURCE_VIDEO = "remotion-video/public/My Movie 360.mp4"
TRAILER_VIDEO = "video_with_hook.mp4"
ASS_FILE = "subtitles.ass"
OUTPUT_VIDEO = "remotion-video/out/final_viral_4k.mp4"
SHORT_OUTPUT_DIR = "remotion-video/out/shorts"
COMPILED_SHORT_OUTPUT = os.path.join(SHORT_OUTPUT_DIR, "compiled_viral_short.mp4")

# Single-word fillers & conversational crutches
FILLER_WORDS = {"UM", "UH", "AH", "ER", "MM", "MHM", "LIKE", "BASICALLY", "ACTUALLY", "SO"}

# Multi-word filler phrases to strip out of captions
FILLER_PHRASES = {"FOR EXAMPLE", "AS WELL", "YOU KNOW", "I MEAN", "SORT OF", "KIND OF"}

HIGHLIGHT_KEYWORDS = {
    "SECRET", "HACK", "BEST", "AMAZING", "CHANGE", "FAST", 
    "MONEY", "POWER", "WIN", "STOP", "NEVER", "ALWAYS", "CREATE", "FREE", "EASY", "NOW", "HUGE", "INSANE"
}
# =======================================================

def remove_dead_silence(input_path, output_path="remotion-video/public/My Movie 360_nosilence.mp4"):
    print(f"\n[Silence Removal] Detecting and cutting silent sections using silencedetect...")
    if not os.path.exists(input_path):
        print(f"⚠️ Error: Could not find video at '{input_path}'.")
        return input_path

    # Step 1: Run silencedetect pass to find quiet sections (< -30dB for >= 0.5s)
    cmd_detect = [
        "ffmpeg", "-hide_banner", "-nostats", "-vn", "-i", input_path,
        "-af", "silencedetect=noise=-30dB:d=0.5",
        "-f", "null", "-"
    ]
    result = subprocess.run(cmd_detect, stderr=subprocess.PIPE, text=True)
    log = result.stderr

    starts = [float(x) for x in re.findall(r"silence_start: (-?[\d.]+)", log)]
    ends = [float(x) for x in re.findall(r"silence_end: (-?[\d.]+)", log)]

    # Get total video duration via ffprobe
    probe_cmd = subprocess.run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration", 
        "-of", "default=noprint_wrappers=1:nokey=1", input_path
    ], stdout=subprocess.PIPE, text=True)
    try:
        total_duration = float(probe_cmd.stdout.strip())
    except Exception:
        total_duration = 0.0

    if not starts:
        print("-> No silences found, keeping original video.")
        return input_path

    if len(ends) < len(starts):
        ends.append(total_duration)

    # Step 2: Invert silence ranges into "keep" (speech) intervals with a small 50ms padding
    keep = []
    cursor = 0.0
    pad = 0.05 
    for s, e in zip(starts, ends):
        if s + pad > cursor:
            keep.append((cursor, max(0.0, s + pad)))
        cursor = max(0.0, e - pad)
    
    if cursor < total_duration:
        keep.append((cursor, total_duration))

    if not keep:
        return input_path

    # Build active speech timestamp expression for FFmpeg select filters
    expr = "+".join(f"between(t,{a:.3f},{b:.3f})" for a, b in keep)
    print(f"-> Cutting out dead air across {len(keep)} active speech segments...")

    # Step 3: Render video and audio streams simultaneously, cutting out the gaps
    v_filter = f"select='{expr}',setpts=N/30/TB"
    a_filter = f"aselect='{expr}',asetpts=N/SR/TB"

    cmd_render = [
        "ffmpeg", "-y", "-i", input_path,
        "-vf", v_filter,
        "-af", a_filter,
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "192k",
        output_path
    ]
    subprocess.run(cmd_render, check=True)
    print(f"-> Silence-free video successfully cut and saved: {output_path}")
    return output_path

def should_filter_out(word_str):
    cleaned = word_str.strip().upper()
    if not cleaned or cleaned in FILLER_WORDS:
        return True
    if any(c in cleaned for c in ["[", "]", "(", ")"]):
        return True
    return False

def clean_chunk_words(chunk):
    filtered = [w for w in chunk if 'word' in w and not should_filter_out(w['word'])]
    if len(filtered) >= 2:
        i = 0
        cleaned_phrase_pass = []
        while i < len(filtered):
            if i < len(filtered) - 1:
                two_word = f"{filtered[i]['word'].strip().upper()} {filtered[i+1]['word'].strip().upper()}"
                if two_word in FILLER_PHRASES:
                    i += 2
                    continue
            cleaned_phrase_pass.append(filtered[i])
            i += 1
        return cleaned_phrase_pass
    return filtered

def format_ass_time(seconds: float) -> str:
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centisecs = int(round((seconds - int(seconds)) * 100))
    if centisecs >= 100:
        centisecs = 99
    return f"{hrs}:{mins:02d}:{secs:02d}.{centisecs:02d}"

def whisperx_transcribe(video_path):
    print(f"\n[WhisperX] Transcribing & aligning '{video_path}'...")
    if not os.path.exists(video_path):
        print(f"⚠️ Error: Could not find video at '{video_path}'.")
        return None

    device = "cpu"
    audio = whisperx.load_audio(video_path)
    
    model = whisperx.load_model("base", device, compute_type="int8")
    result = model.transcribe(audio, batch_size=16)

    print(f"-> Running forced phoneme alignment...")
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    
    segments = result.get('segments', [])
    print(f"-> Alignment complete. Found {len(segments)} synchronized segments.")
    return segments

def extract_frame_at_time(video_path, timestamp, output_image_path):
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{timestamp:.3f}",
        "-i", video_path,
        "-vframes", "1",
        "-q:v", "2",
        output_image_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def extract_and_compile_viral_short(segments, source_video):
    print(f"\n[Shorts Pipeline] Scanning video for global highlights (BART + CLIP)...")
    os.makedirs(SHORT_OUTPUT_DIR, exist_ok=True)
    
    text_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=-1)
    text_labels = ["high-attention viral hook", "informative core content", "casual filler"]
    
    vision_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    vision_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    visual_queries = ["an exciting visual action scene", "a close-up product reveal or interesting object", "a dramatic or expressive moment"]

    scored_segments = []
    temp_frame_path = "temp_eval_frame.jpg"

    for seg in segments:
        text = seg.get('text', '').strip()
        duration = seg['end'] - seg['start']
        
        if not (2.5 <= duration <= 8.0):
            continue

        text_score = 0.5
        try:
            text_pred = text_classifier(text, text_labels)
            text_score = text_pred['scores'][text_pred['labels'].index("high-attention viral hook")]
        except Exception:
            pass

        mid_time = seg['start'] + (duration / 2.0)
        visual_score = 0.5 
        try:
            extract_frame_at_time(source_video, mid_time, temp_frame_path)
            if os.path.exists(temp_frame_path):
                image = Image.open(temp_frame_path)
                inputs = vision_processor(text=visual_queries, images=image, return_tensors="pt", padding=True)
                outputs = vision_model(**inputs)
                visual_score = outputs.logits_per_image.softmax(dim=1).max().item()
                image.close()
        except Exception:
            pass

        combined_score = (text_score * 0.6) + (visual_score * 0.4)
        scored_segments.append((combined_score, seg))

    if os.path.exists(temp_frame_path):
        os.remove(temp_frame_path)

    scored_segments.sort(key=lambda x: x[0], reverse=True)

    chosen_segments = []
    accumulated_duration = 0.0
    target_duration = 60.0

    for score, seg in scored_segments:
        duration = seg['end'] - seg['start']
        if accumulated_duration + duration > target_duration and len(chosen_segments) > 0:
            continue
            
        overlap = False
        for existing in chosen_segments:
            if not (seg['end'] < existing['start'] or seg['start'] > existing['end']):
                overlap = True
                break
                
        if not overlap:
            chosen_segments.append(seg)
            accumulated_duration += duration
            
        if accumulated_duration >= target_duration:
            break

    chosen_segments.sort(key=lambda x: x['start'])
    print(f"🎯 Compiled Short Highlights Selected: {len(chosen_segments)} clips ({accumulated_duration:.1f}s total)")

    if not chosen_segments:
        print("⚠️ No highlights found for short compilation.")
        return

    temp_short_ass = "temp_compiled_short.ass"
    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: SpringShorts,Arial Black,70,&H00FFFFFF,&H0000FFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,5,3,2,50,50,300,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events = []
    current_compiled_time = 0.0

    for chosen in chosen_segments:
        c_start = chosen['start']
        c_end = chosen['end']
        dur = c_end - c_start

        for segment in segments:
            if segment['start'] >= c_start and segment['end'] <= c_end:
                words = segment.get('words', [])
                if not words: continue
                
                clean_words = clean_chunk_words(words)
                if not clean_words: continue

                chunk_size = 3
                for i in range(0, len(clean_words), chunk_size):
                    chunk = clean_words[i:i + chunk_size]
                    if not chunk or 'start' not in chunk[0] or 'end' not in chunk[-1]: continue
                    
                    offset = current_compiled_time - c_start
                    line_start = chunk[0]['start'] + offset
                    line_end = chunk[-1]['end'] + offset + 0.15
                    
                    line_text = r"{\fad(50,50)}"
                    for w in chunk:
                        w_text = w['word'].strip().upper()
                        is_keyword = w_text in HIGHLIGHT_KEYWORDS or len(w_text) > 6
                        if is_keyword:
                            spring_tag = r"{\c&H0000FFFF&\fscx140\fscy140\t(0,80,105\fscx100\fscy100)}"
                            line_text += f"{spring_tag}{w_text}" + r"{\c&H00FFFFFF&\fscx100\fscy100} "
                        else:
                            line_text += r"{\c&H00FFFFFF&\fscx100\fscy100}" + w_text + " "
                    
                    events.append(f"Dialogue: 0,{format_ass_time(line_start)},{format_ass_time(line_end)},SpringShorts,,0,0,0,,{line_text.strip()}")

        current_compiled_time += dur

    with open(temp_short_ass, "w", encoding="utf-8") as f:
        f.write(header + "\n".join(events))

    clip_files = []
    for idx, seg in enumerate(chosen_segments):
        start = max(0, seg['start'] - 0.05)
        end = seg['end'] + 0.05
        clip_name = f"temp_highlight_{idx}.mp4"
        vf = f"trim=start={start}:end={end},setpts=PTS-STARTPTS,scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920:(iw-1080)/2:(ih-1920)/2"
        af = f"atrim=start={start}:end={end},asetpts=PTS-STARTPTS"
        
        subprocess.run(["ffmpeg", "-y", "-i", source_video, "-vf", vf, "-af", af, "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23", "-c:a", "aac", "-b:a", "192k", clip_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        clip_files.append(clip_name)

    concat_list = "concat_highlights.txt"
    with open(concat_list, "w", encoding="utf-8") as f:
        for clip in clip_files: f.write(f"file '{clip}'\n")

    raw_concat = "temp_raw_compiled.mp4"
    subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_list, "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23", "-c:a", "aac", "-b:a", "192k", raw_concat], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    subprocess.run(["ffmpeg", "-y", "-i", raw_concat, "-vf", f"ass={temp_short_ass}", "-af", "loudnorm=I=-12:TP=-1.5:LRA=11", "-c:v", "libx264", "-crf", "23", "-preset", "fast", "-c:a", "aac", "-b:a", "192k", COMPILED_SHORT_OUTPUT], check=True)

    for clip in clip_files:
        if os.path.exists(clip): os.remove(clip)
    if os.path.exists(concat_list): os.remove(concat_list)
    if os.path.exists(raw_concat): os.remove(raw_concat)
    if os.path.exists(temp_short_ass): os.remove(temp_short_ass)

    print(f"✨ Compiled Viral Short successfully generated: {COMPILED_SHORT_OUTPUT}")

def ai_multimodal_extract_and_prepend_hook(segments, source_video):
    print(f"\n[Multimodal AI Hook Finder] Generating main video intro hook...")
    text_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=-1)
    text_labels = ["high-attention viral hook", "boring setup", "casual filler speech"]
    
    vision_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    vision_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    visual_queries = ["an exciting visual action scene", "a close-up product reveal or interesting object", "a dramatic or expressive moment"]

    scored_segments = []
    temp_frame_path = "temp_eval_frame.jpg"

    for seg in segments:
        text = seg.get('text', '').strip()
        duration = seg['end'] - seg['start']
        if not (2.0 <= duration <= 7.0): continue

        text_score = text_classifier(text, text_labels)['scores'][0]
        mid_time = seg['start'] + (duration / 2.0)
        visual_score = 0.5 
        try:
            extract_frame_at_time(source_video, mid_time, temp_frame_path)
            if os.path.exists(temp_frame_path):
                image = Image.open(temp_frame_path)
                inputs = vision_processor(text=visual_queries, images=image, return_tensors="pt", padding=True)
                visual_score = vision_model(**inputs).logits_per_image.softmax(dim=1).max().item()
                image.close()
        except Exception: pass

        scored_segments.append(((text_score * 0.5) + (visual_score * 0.5), seg))

    if os.path.exists(temp_frame_path): os.remove(temp_frame_path)
    scored_segments.sort(key=lambda x: x[0], reverse=True)

    top_segments = []
    for score, seg in scored_segments:
        if len(top_segments) >= 5: break
        if not any(not (seg['end'] < ex['start'] or seg['start'] > ex['end']) for ex in top_segments):
            top_segments.append(seg)

    if not top_segments: return source_video
    top_segments.sort(key=lambda x: x['start'])

    clip_files = []
    for idx, seg in enumerate(top_segments):
        hook_start = max(0, seg['start'] - 0.1)
        hook_end = min(seg['end'] + 0.1, hook_start + 5.0)
        clip_name = f"temp_clip_{idx}.mp4"
        subprocess.run(["ffmpeg", "-y", "-ss", f"{hook_start:.3f}", "-i", source_video, "-t", f"{hook_end - hook_start:.3f}", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23", "-c:a", "aac", "-b:a", "128k", clip_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        clip_files.append(clip_name)

    montage_list = "montage_list.txt"
    with open(montage_list, "w", encoding="utf-8") as f:
        for clip in clip_files: f.write(f"file '{clip}'\n")

    raw_montage = "temp_montage.mp4"
    watermark_filter = "drawtext=text='In this video you will see...':fontcolor=white:fontsize=150:borderw=2:bordercolor=white:x=(w-tw)/2:y=150:box=0"
    subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", montage_list, "-vf", watermark_filter, "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23", "-c:a", "aac", "-b:a", "128k", raw_montage], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    probe_cmd = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", raw_montage], stdout=subprocess.PIPE, text=True)
    montage_duration = float(probe_cmd.stdout.strip())
    fade_duration = 0.5 

    filter_complex = f"[0:v][1:v]xfade=transition=fade:duration={fade_duration}:offset={montage_duration - fade_duration}[v];[0:a][1:a]acrossfade=d={fade_duration}[a]"
    subprocess.run(["ffmpeg", "-y", "-i", raw_montage, "-i", source_video, "-filter_complex", filter_complex, "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23", "-c:a", "aac", "-b:a", "128k", "-fps_mode", "cfr", TRAILER_VIDEO], check=True)

    for clip in clip_files:
        if os.path.exists(clip): os.remove(clip)
    if os.path.exists(montage_list): os.remove(montage_list)
    if os.path.exists(raw_montage): os.remove(raw_montage)

    return TRAILER_VIDEO

def generate_4k_ass(segments, output_ass=ASS_FILE):
    print(f"\n[Subtitles] Generating 4K ASS kinetic typography script...")
    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 3840
PlayResY: 2160

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: ProCaption4K,Arial Black,130,&H00FFFFFF,&H0000FFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,8,6,2,200,200,220,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events = []
    for segment in segments:
        words = segment.get('words', [])
        if not words: continue
        
        clean_words = clean_chunk_words(words)
        if not clean_words: continue
            
        for i in range(0, len(clean_words), 3):
            chunk = clean_words[i:i + 3]
            if not chunk or 'start' not in chunk[0] or 'end' not in chunk[-1]: continue
            
            line_text = r"{\fad(80,80)}"
            for w in chunk:
                w_text = w['word'].strip().upper()
                if w_text in HIGHLIGHT_KEYWORDS or len(w_text) > 6:
                    line_text += r"{\c&H0000FFFF&\fscx115\fscy115}" + w_text + r"{\c&H00FFFFFF&\fscx100\fscy100} "
                else:
                    line_text += r"{\c&H00FFFFFF&\fscx100\fscy100}" + w_text + " "
            
            events.append(f"Dialogue: 0,{format_ass_time(chunk[0]['start'])},{format_ass_time(chunk[-1]['end'] + 0.15)},ProCaption4K,,0,0,0,,{line_text.strip()}")

    with open(output_ass, "w", encoding="utf-8") as f:
        f.write(header + "\n".join(events))

def render_final_4k(active_video):
    print(f"\n[Final Render] Rendering master 4K video...")
    os.makedirs("remotion-video/out", exist_ok=True)
    
    video_filter = "scale=3840:2160:force_original_aspect_ratio=decrease,pad=3840:2160:(ow-iw)/2:(oh-ih)/2,zoompan=z='min(zoom+0.0008,1.12)':x='iw/2-(iw/zoom/2)':y='ih/2-(iw/zoom/2)':d=1:s=3840x2160:fps=30," + f"ass={ASS_FILE}," + "fade=in:st=0:d=1"
    
    cmd = ["ffmpeg", "-y", "-i", active_video, "-vf", video_filter, "-af", "loudnorm=I=-12:TP=-1.5:LRA=11", "-c:v", "libx264", "-crf", "23", "-preset", "fast", "-c:a", "aac", "-b:a", "192k", OUTPUT_VIDEO]
    subprocess.run(cmd, check=True)
    print(f"\n✨ Master 4K video rendered: {OUTPUT_VIDEO}")

if __name__ == "__main__":
    # STEP 0: Physical dead-silence removal from source video
    cleaned_source = remove_dead_silence(SOURCE_VIDEO)
    
    # Transcribe and align the newly cut, silence-free video stream
    initial_segments = whisperx_transcribe(cleaned_source)
    
    if initial_segments:
        # STEP 1: Extract and compile 9:16 viral short from highlights
        extract_and_compile_viral_short(initial_segments, cleaned_source)
        
        # STEP 2: Process main video hook injection & full 4K rendering pipeline
        processed_video = ai_multimodal_extract_and_prepend_hook(initial_segments, cleaned_source)
        final_segments = whisperx_transcribe(processed_video)
        
        if final_segments:
            generate_4k_ass(final_segments)
            render_final_4k(processed_video)