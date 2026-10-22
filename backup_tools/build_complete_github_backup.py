"""Create an immutable, credential-sanitized snapshot; never mutate source files."""
from __future__ import annotations

import argparse
import base64
import collections
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import tarfile
import time
import zipfile

PATTERNS = [
    ("service_key", re.compile(rb"(?:sk-[A-Fa-f0-9]{24,128}|sk-(?:proj-|svcacct-|ant-)[A-Za-z0-9_-]{20,}|sk-[A-Za-z0-9]{48,128}|gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{40,}|AIza[0-9A-Za-z_-]{35}|AKIA[0-9A-Z]{16})(?![A-Za-z0-9])")),
    ("private_key", re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA |ENCRYPTED )?PRIVATE KEY-----.*?-----END (?:RSA |EC |OPENSSH |DSA |ENCRYPTED )?PRIVATE KEY-----", re.S)),
    ("bearer", re.compile(rb"(?i)(\bbearer\s+)[A-Za-z0-9._~+/-]{20,}={0,2}")),
    ("credential_assignment", re.compile(rb'''(?i)((?:api[_-]?key|access[_-]?token|refresh[_-]?token|client[_-]?secret|authorization|password)["']?\s*[:=]\s*["'])[A-Za-z0-9._~+/-]{16,}(["'])''')),
    ("jwt", re.compile(rb"(?<![A-Za-z0-9])eyJ[A-Za-z0-9_-]{15,}\.eyJ[A-Za-z0-9_-]{15,}\.[A-Za-z0-9_-]{15,}")),
]
SECRET_NAMES = {".env", ".netrc", ".git-credentials", ".npmrc", "auth.json", "id_rsa", "id_ed25519", "id_ecdsa", "credentials.json"}
SECRET_SUFFIXES = {".dpapi", ".pfx", ".p12"}


class Sanitizer:
    def __init__(self):
        self.counts = collections.Counter()
        self.notices = []

    def excluded(self, name):
        p = Path(name)
        lower = [x.lower() for x in p.parts]
        return (".git" in lower or "secrets" in lower or ".sandbox-secrets" in lower or "auths" in lower or "credentials" in lower
                or p.name.lower() in SECRET_NAMES or p.name.lower().startswith(".env.")
                or p.suffix.lower() in SECRET_SUFFIXES)

    def raw(self, data, label):
        lower = None
        for kind, pattern in PATTERNS:
            if kind == "service_key" and not any(x in data for x in (b"sk-", b"ghp_", b"gho_", b"ghu_", b"ghs_", b"ghr_", b"github_pat_", b"AIza", b"AKIA")):
                continue
            if kind == "private_key" and b"PRIVATE KEY-----" not in data:
                continue
            if kind == "jwt" and b"eyJ" not in data:
                continue
            if kind in ("bearer", "credential_assignment"):
                if lower is None:
                    lower = data.lower()
                if kind == "bearer" and b"bearer" not in lower:
                    continue
                if kind == "credential_assignment" and not any(x in lower for x in (b"api_key", b"api-key", b"apikey", b"access_token", b"access-token", b"accesstoken", b"refresh_token", b"refresh-token", b"client_secret", b"client-secret", b"authorization", b"password")):
                    continue
            def replace(m):
                self.counts[kind] += 1
                if kind == "bearer":
                    return m.group(1) + b"[REDACTED]"
                if kind == "credential_assignment":
                    return m.group(1) + b"[REDACTED]" + m.group(2)
                return b"[REDACTED]"
            data = pattern.sub(replace, data)
        # Also cover UTF-16 encoded text without changing unrelated binaries.
        if data[:2] in (b"\xff\xfe", b"\xfe\xff"):
            enc = "utf-16"
            try:
                decoded = data.decode(enc).encode("utf-8")
                cleaned = self.raw(decoded, label + "::utf16")
                if cleaned != decoded:
                    data = cleaned.decode("utf-8").encode(enc)
            except UnicodeError:
                pass
        return data

    def clean(self, data, label, depth=0):
        if depth > 8:
            raise RuntimeError("Nested archive depth exceeds safety limit: " + label)
        lower = label.lower()
        if data[:4] in (b"PK\x03\x04", b"PK\x05\x06"):
            try:
                with zipfile.ZipFile(io.BytesIO(data)) as zin:
                    changed = False
                    records = []
                    for info in zin.infolist():
                        if self.excluded(info.filename):
                            self.counts["embedded_credential_or_git_excluded"] += 1
                            changed = True
                            continue
                        if info.flag_bits & 1:
                            raise RuntimeError("Encrypted nested archive cannot be credential-scanned: " + label)
                        original = zin.read(info)
                        updated = self.clean(original, label + "::" + info.filename, depth + 1) if not info.is_dir() else original
                        changed |= updated != original
                        records.append((info, updated))
                    if changed:
                        out = io.BytesIO()
                        with zipfile.ZipFile(out, "w", allowZip64=True) as zout:
                            for info, content in records:
                                zout.writestr(info, content)
                        return out.getvalue()
                    return data
            except (zipfile.BadZipFile, NotImplementedError, UnicodeError, EOFError) as exc:
                self.notices.append({"path": label, "note": "invalid_archive_fixture_raw_scanned", "error": type(exc).__name__})
        elif data[:2] == b"\x1f\x8b":
            try:
                expanded = gzip.decompress(data)
                updated = self.clean(expanded, label + "::gzip", depth + 1)
                return gzip.compress(updated, compresslevel=6, mtime=0) if updated != expanded else data
            except (OSError, EOFError):
                self.notices.append({"path": label, "note": "invalid_gzip_raw_scanned"})
        elif lower.endswith((".tar", ".tgz::gzip", ".tar.gz::gzip")):
            try:
                records = []
                changed = False
                with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as tin:
                    for info in tin:
                        if self.excluded(info.name):
                            self.counts["embedded_credential_or_git_excluded"] += 1
                            changed = True
                            continue
                        stream = tin.extractfile(info) if info.isfile() else None
                        original = stream.read() if stream else b""
                        updated = self.clean(original, label + "::" + info.name, depth+1) if stream else original
                        changed |= original != updated
                        records.append((info, updated))
                if changed:
                    out = io.BytesIO()
                    with tarfile.open(fileobj=out, mode="w") as tout:
                        for info, content in records:
                            info.size = len(content) if info.isfile() else info.size
                            tout.addfile(info, io.BytesIO(content) if info.isfile() else None)
                    return out.getvalue()
                return data
            except (tarfile.TarError, OSError, UnicodeError):
                self.notices.append({"path": label, "note": "invalid_tar_fixture_raw_scanned"})
        return self.raw(data, label)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--session", required=True)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    source, stage = Path(args.source), Path(args.stage)
    stage.mkdir(parents=True, exist_ok=args.resume)
    san = Sanitizer()
    excluded = []
    files = []
    for parent, dirs, names in os.walk(source):
        keep=[]
        for d in sorted(dirs):
            rel=(Path(parent)/d).relative_to(source)
            if san.excluded(str(rel)):
                excluded.append({"path": rel.as_posix()+"/", "reason": "credential_storage_or_git_metadata"})
            else: keep.append(d)
        dirs[:]=keep
        for name in sorted(names):
            p=Path(parent)/name
            rel=p.relative_to(source)
            if san.excluded(str(rel)):
                excluded.append({"path": rel.as_posix(), "reason": "credential_file"})
            else: files.append((p, rel))
    print(json.dumps({"phase":"inventory","files":len(files),"source_bytes":sum(p.stat().st_size for p,r in files),"excluded_paths":len(excluded)}),flush=True)
    manifest=stage/"BACKUP_MANIFEST.jsonl"
    resume_rows = {}
    if args.resume and manifest.exists():
        with manifest.open(encoding="utf-8") as prior:
            for line in prior:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                target = stage / row["path"]
                if target.is_file() and target.stat().st_size == row["export_bytes"]:
                    resume_rows[row["path"]] = row
    totals=collections.Counter()
    started=time.monotonic()
    with manifest.open("w",encoding="utf-8") as mf:
        for i,(p,rel) in enumerate(files,1):
            resumed = resume_rows.get("workspace/" + rel.as_posix())
            if resumed and resumed["source_bytes"] == p.stat().st_size:
                mf.write(json.dumps(resumed,ensure_ascii=False)+"\n")
                totals["files"]+=1;totals["source_bytes"]+=resumed["source_bytes"];totals["export_bytes"]+=resumed["export_bytes"]
                san.counts.update(resumed.get("redactions",{}))
                if resumed.get("redactions"):totals["modified_files"]+=1
                if i%10000==0:print(json.dumps({"phase":"resume","reused_files":i}),flush=True)
                continue
            before=san.counts.copy()
            data=p.read_bytes()
            original_sha=hashlib.sha256(data).hexdigest()
            cleaned=san.clean(data, rel.as_posix())
            q=stage/"workspace"/rel
            q.parent.mkdir(parents=True,exist_ok=True)
            q.write_bytes(cleaned)
            shutil.copystat(p,q)
            changes=san.counts-before
            row={"path":"workspace/"+rel.as_posix(),"source_bytes":len(data),"export_bytes":len(cleaned),"source_sha256":original_sha,"sha256":hashlib.sha256(cleaned).hexdigest(),"redactions":dict(changes)}
            mf.write(json.dumps(row,ensure_ascii=False)+"\n")
            mf.flush()
            totals["files"]+=1; totals["source_bytes"]+=len(data); totals["export_bytes"]+=len(cleaned)
            if changes:totals["modified_files"]+=1
            if i%1000==0 or time.monotonic()-started>30:
                print(json.dumps({"phase":"sanitize","done":i,"total":len(files),"bytes":totals["source_bytes"],"redactions":dict(san.counts)}),flush=True)
                started=time.monotonic()
    chat=stage/"conversation"
    chat.mkdir()
    messages=0
    with Path(args.session).open(encoding="utf-8") as src, (chat/"messages.jsonl").open("w",encoding="utf-8") as out, (chat/"CHAT_HISTORY.md").open("w",encoding="utf-8") as md:
        md.write("# 项目原始对话记录（密钥已脱敏）\n\n本文件导出本机保存的用户消息、助手最终答复和过程更新。不含系统指令、内部推理或工具执行记录。模型调用缓存与实验日志位于完整工作区。\n\n")
        for line in src:
            try: obj=json.loads(line)
            except ValueError: continue
            pay=obj.get("payload",{})
            if obj.get("type")!="response_item" or pay.get("type")!="message" or pay.get("role") not in ("user","assistant"):continue
            if pay.get("role")=="assistant" and pay.get("phase") not in (None,"commentary","final_answer","final"):continue
            # Exclude injected environment/context records that are not human dialogue.
            texts=[];attachments=[]
            for block in pay.get("content",[]):
                if not isinstance(block,dict):continue
                if block.get("type") in ("input_text","output_text","text"):
                    texts.append(block.get("text",""))
                elif block.get("type")=="input_image":
                    image_url=block.get("image_url","")
                    if isinstance(image_url,dict):image_url=image_url.get("url","")
                    if isinstance(image_url,str) and image_url.startswith("data:") and ";base64," in image_url:
                        head,b64=image_url.split(",",1)
                        ext="jpg" if "jpeg" in head else "png"
                        name=f"attachment_{messages+1:05d}_{len(attachments)+1}.{ext}"
                        (chat/name).write_bytes(base64.b64decode(b64))
                        attachments.append(name)
                    else:attachments.append("[image attachment referenced in original message]")
            body="\n\n".join(texts)
            if pay.get("role")=="user" and body.lstrip().startswith("<environment_context>"):continue
            body=san.raw(body.encode("utf-8"),"conversation").decode("utf-8")
            messages+=1
            row={"index":messages,"timestamp":obj.get("timestamp"),"role":pay["role"],"phase":pay.get("phase"),"text":body,"attachments":attachments}
            out.write(json.dumps(row,ensure_ascii=False)+"\n")
            md.write(f"## {messages}. {row['role']} — {row['timestamp']}\n\n{body}\n\n")
            for name in attachments:
                md.write(f"![附件]({name})\n\n" if not name.startswith("[") else name+"\n\n")
    (stage/"EXCLUDED_CREDENTIALS.json").write_text(json.dumps(excluded,ensure_ascii=False,indent=2),encoding="utf-8")
    (stage/"SANITIZATION_NOTES.json").write_text(json.dumps(san.notices,ensure_ascii=False,indent=2),encoding="utf-8")
    summary={"snapshot_date":"2026-10-22","scope":"complete existing project workspace and saved user-visible conversation","totals":dict(totals),"conversation_messages":messages,"redactions":dict(san.counts),"excluded_path_count":len(excluded),"archive_fixture_notes":len(san.notices)}
    (stage/"BACKUP_SUMMARY.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"phase":"sanitized_snapshot_complete",**summary}),flush=True)


if __name__=="__main__":main()
