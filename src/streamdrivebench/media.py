"""Recursive video inventory and causal 1-Hz sampling from decoded presentation times."""
import hashlib
import math
from pathlib import Path
import av
from PIL import Image, ImageOps
from .common import dump_json, load_json, file_hash

EXTENSIONS={'.mp4','.mov','.mkv','.avi','.webm','.m4v','.mts','.m2ts'}
IMAGE_SIZE=(910,512)


def discover_clips(root):
    root=Path(root).resolve()
    if not root.is_dir():raise ValueError(f'Video folder does not exist: {root}')
    files=sorted(p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS)
    print(f'VIDEO_INVENTORY found={len(files)} root={root}',flush=True)
    if not files:raise ValueError('No supported videos found')
    clips=[]
    for path in files:
        name=path.relative_to(root).as_posix()
        with av.open(str(path)) as c:
            if not c.streams.video:raise ValueError(f'No video stream: {name}')
            s=c.streams.video[0]
            duration=float(s.duration*s.time_base) if s.duration is not None else None
            clips.append(dict(file=name,episode_id='video-'+hashlib.sha256(name.encode()).hexdigest()[:20],
                sequence=path.parent.relative_to(root).as_posix(),split='unspecified',
                frame_count=s.frames or None,duration_seconds=duration,
                source_width=s.width,source_height=s.height))
    return clips


def prepare_episode(clip, source, out):
    root=Path(out)/'source_1fps';path=Path(source)/clip['file'];eid=clip['episode_id']
    marker=root/'prepared'/f'{eid}.json'
    if marker.exists():
        saved=load_json(marker)
        if saved['source_sha256']!=clip['sha256']:raise ValueError('Source video changed')
        for f in saved['episode']['frames']:
            if file_hash(root/f['image_paths']['CAM_FRONT'])!=saved['image_hashes'][str(f['frame_index'])]:
                raise ValueError('Prepared image changed')
        return saved['episode']
    rows=[];hashes={};next_t=0;previous=None;previous_time=None;previous_index=None;origin=None
    last=-1.;last_span=0.;fallback=0.
    def save(t,frame,pts,index):
        if frame is None or pts>t:raise ValueError('Missing causal frame at integer second')
        rel=f'frames/{eid}/{t:06d}.jpg';dest=root/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        im=ImageOps.pad(frame.to_image().convert('RGB'),IMAGE_SIZE,method=Image.Resampling.LANCZOS,color='black')
        temp=dest.with_suffix('.tmp');im.save(temp,format='JPEG',quality=90);temp.replace(dest)
        hashes[str(t)]=file_hash(dest)
        rows.append(dict(frame_index=t,timestamp=t*1000000,time_sec=t,source_video_time_sec=pts,
            source_clip_frame_index=index,source_frame_index=index,image_age_sec=t-pts,
            observation_valid=0<=t-pts<=0.5,image_paths={'CAM_FRONT':rel},
            camera_data={'CAM_FRONT':dict(width=910,height=512)}))
    with av.open(str(path)) as container:
        stream=container.streams.video[0];stream.thread_count=2
        fallback=1/float(stream.average_rate) if stream.average_rate else 0.
        for index,frame in enumerate(container.decode(stream)):
            if frame.pts is None:raise ValueError('Decoded frame lacks presentation timestamp')
            absolute=float(frame.pts*frame.time_base)
            if origin is None:origin=absolute
            pts=round(absolute-origin,9)
            if pts<=last:raise ValueError('Nonmonotonic decoded presentation timestamps')
            while next_t<pts:
                save(next_t,previous,previous_time,previous_index);next_t+=1
            last_span=pts-last if last>=0 else fallback
            span=float(frame.duration*frame.time_base) if getattr(frame,'duration',0) else last_span or fallback
            previous,previous_time,previous_index,last=frame,pts,index,pts
        if previous is None:raise ValueError('Empty/undecodable video')
        if span<=0:raise ValueError('Cannot determine final frame duration')
        duration=round(last+span,9)
        while next_t<duration:
            save(next_t,previous,previous_time,previous_index);next_t+=1
    ep=dict(episode_id=eid,split='unspecified',fps=1,num_frames=len(rows),duration_sec=duration,
        source_video=str(path.resolve()),source_sequence=clip['sequence'],source_camera='front',
        source_first_frame=0,source_last_frame=previous_index,source_width=clip['source_width'],
        source_height=clip['source_height'],source_pts_origin_sec=origin,frames=rows,
        unavailable_seconds=[f['frame_index'] for f in rows if not f['observation_valid']])
    dump_json(dict(source_sha256=clip['sha256'],episode=ep,image_hashes=hashes),marker)
    print(f'PREPARED {clip["file"]} seconds={duration:.3f} observations={len(rows)}',flush=True)
    return ep


class Dataset:
    def __init__(self,root,episodes):
        self.root=Path(root);self.episodes=episodes;self.episode_by_id={e['episode_id']:e for e in episodes};self.image_size=list(IMAGE_SIZE)
    def image_path(self,eid,index):return self.root/self.episode_by_id[eid]['frames'][index]['image_paths']['CAM_FRONT']
    def valid(self,eid,frames):
        ep=self.episode_by_id[eid]
        return all(0<=f<len(ep['frames']) and 0<=f-ep['frames'][f]['source_video_time_sec']<=0.5 for f in frames)
