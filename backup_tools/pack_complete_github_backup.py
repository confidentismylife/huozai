"""Pack sanitized snapshot into SHA-256 verified, GitHub-sized ZIP volumes."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import time
import zipfile


class Volumes(io.RawIOBase):
    def __init__(self, directory, limit):
        self.directory=Path(directory)
        self.directory.mkdir(parents=True,exist_ok=True)
        self.limit=limit;self.position=0;self.handle=None;self.part_size=0
        self.parts=[];self.whole=hashlib.sha256();self.digest=None

    def writable(self):return True
    def seekable(self):return False
    def tell(self):return self.position

    def finish_part(self):
        if self.handle:
            self.handle.close()
            self.parts.append({"name":self.handle.name.split("\\")[-1].split("/")[-1],"bytes":self.part_size,"sha256":self.digest.hexdigest()})
            self.handle=None

    def write(self,data):
        total=len(data);view=memoryview(data)
        while view:
            if self.handle is None:
                p=self.directory/f"workspace_backup.zip.part{len(self.parts)+1:04d}"
                self.handle=p.open("wb");self.digest=hashlib.sha256();self.part_size=0
            size=min(len(view),self.limit-self.part_size)
            b=view[:size];self.handle.write(b);self.digest.update(b);self.whole.update(b)
            self.part_size+=size;self.position+=size;view=view[size:]
            if self.part_size==self.limit:self.finish_part()
        return total

    def finish(self):
        self.finish_part()
        return {"format":"split ZIP64 byte stream","bytes":self.position,"sha256":self.whole.hexdigest(),"parts":self.parts}


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--stage",required=True);ap.add_argument("--output",required=True)
    args=ap.parse_args();stage=Path(args.stage);output=Path(args.output)
    files=sorted(p for p in stage.rglob("*") if p.is_file())
    v=Volumes(output,45*1024*1024);start=time.monotonic()
    with zipfile.ZipFile(v,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True,strict_timestamps=False) as z:
        for i,p in enumerate(files,1):
            z.write(p,p.relative_to(stage).as_posix())
            if i%1000==0 or time.monotonic()-start>30:
                print(json.dumps({"phase":"compress","done":i,"total":len(files),"compressed_bytes":v.tell(),"closed_parts":len(v.parts)}),flush=True)
                start=time.monotonic()
    info=v.finish();info["files"]=len(files)
    (output/"VOLUMES.json").write_text(json.dumps(info,indent=2),encoding="utf-8")
    (output/"SHA256SUMS.txt").write_text("".join(f"{p['sha256']}  {p['name']}\n" for p in info["parts"]),encoding="ascii")
    print(json.dumps({"phase":"compression_complete","bytes":info['bytes'],"parts":len(info['parts']),"files":len(files),"sha256":info['sha256']}),flush=True)


if __name__=="__main__":main()
