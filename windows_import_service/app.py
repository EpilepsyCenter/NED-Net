from fastapi import FastAPI, UploadFile, File
from fastapi.responses import ORJSONResponse
from pathlib import Path
import uuid
import orjson

app = FastAPI(default_response_class=ORJSONResponse)

OUTPUT_ROOT = Path(r"Z:\exports\imports")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

@app.get("/status")
def status():
    return {"ok": True}

@app.post("/import")
async def import_adicht(file: UploadFile = File(...)):
    import_id = str(uuid.uuid4())
    out_dir = OUTPUT_ROOT / import_id
    out_dir.mkdir(parents=True, exist_ok=True)

    dst = out_dir / file.filename
    with open(dst, "wb") as f:
        while True:
            chunk = await file.read(8 * 1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    meta = {
        "import_id": import_id,
        "saved_path": str(dst),
        "filename": file.filename,
    }
    (out_dir / "meta.json").write_bytes(orjson.dumps(meta, option=orjson.OPT_INDENT_2))
    return meta