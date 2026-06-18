from pathlib import Path
import sys
import uuid
import requests

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

STATIC_DIR = APP_DIR / "static"
UPLOAD_DIR = APP_DIR / "uploads"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from ml.predict_api import predict_image_bytes
from ml.compare_api import compare_image_bytes

app = FastAPI(title="FloraGuide API")

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def normalize_plant_name(name: str):
    return name.split(" (")[0].strip()


def get_wikipedia_text(plant: str):
    try:
        name = normalize_plant_name(plant).replace(" ", "_")
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{name}"

        r = requests.get(
            url,
            timeout=10,
            headers={"User-Agent": "FloraGuide/1.0"}
        )

        if r.status_code != 200:
            return None

        data = r.json()
        return data.get("extract")

    except Exception:
        return None


def generate_care_from_text(plant: str, wiki_text: str | None):
    if not wiki_text:
        return default_care()

    care_text = f"""
{plant} bitkisinin sağlıklı gelişebilmesi için aşağıdaki önerilere dikkat edilmesi tavsiye edilir.

Sulama:
Toprağı tamamen kupkuru bırakmadan, düzenli aralıklarla kontrollü şekilde sulayın. Fazla suyun saksıda beklememesine dikkat edin.

Işık:
Bitkiyi aydınlık bir ortamda tutun. Gün içinde doğal ışık alması gelişimini destekler. Çok sert ve yakıcı güneş ışığında uzun süre bırakmamaya dikkat edin.

Sıcaklık:
Genel olarak 15 ile 25 derece arasındaki sıcaklık çoğu süs bitkisi için uygundur. Ani sıcaklık değişimlerinden koruyun.

Genel bakım:
Yaprakları zaman zaman kontrol edin. Kuruyan veya zarar gören yaprakları temizlemek bitkinin daha sağlıklı görünmesine yardımcı olur.
"""
    return care_text.strip()


def default_care():
    return """
BAKIM BULUNAMADI!!!

Bitkinin sağlıklı gelişmesi için şu önerilere dikkat edebilirsiniz:

Sulama:
Toprağın üst kısmı hafif kurudukça sulama yapın ve saksıda su birikmemesine dikkat edin.

Işık:
Bitkinizi aydınlık bir ortamda tutun. Doğrudan çok yakıcı güneş yerine dolaylı doğal ışık daha uygundur.

Sıcaklık:
Çoğu ev ve süs bitkisi için 15 ile 25 derece arası uygun kabul edilir.

Genel bakım:
Bitkinin yapraklarını düzenli kontrol edin ve kuruyan yaprakları temizleyin.
""".strip()


@app.get("/")
def root():
    return {"message": "FloraGuide API çalışıyor"}


@app.get("/ui")
def ui():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Sadece JPG veya PNG desteklenir.")

    ext = file.filename.split(".")[-1].lower()

    if ext not in ["jpg", "jpeg", "png"]:
        raise HTTPException(status_code=400, detail="Dosya uzantısı jpg/jpeg/png olmalıdır.")

    filename = f"{uuid.uuid4()}.{ext}"
    file_path = UPLOAD_DIR / filename

    with open(file_path, "wb") as f:
        f.write(await file.read())

    return {
        "message": "Fotoğraf başarıyla yüklendi",
        "filename": filename
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Sadece JPG veya PNG desteklenir.")

    try:
        image_bytes = await file.read()
        prediction = predict_image_bytes(image_bytes)

        return {
            "message": "AI tahmin tamamlandı",
            "plant": prediction["plant"],
            "model": prediction["model"]
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"AI prediction error: {type(e).__name__} - {str(e)}"
        )


@app.post("/identify-care")
async def identify_and_care(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Sadece JPG veya PNG desteklenir.")

    try:
        image_bytes = await file.read()
        prediction = predict_image_bytes(image_bytes)

        plant = prediction["plant"]
        model = prediction["model"]

        if plant != "unknown":
            wiki_text = get_wikipedia_text(plant)
            care_text = generate_care_from_text(plant, wiki_text)
        else:
            care_text = default_care()

        return {
            "message": "Bitki analizi tamamlandı",
            "plant": plant,
            "model": model,
            "care_recommendation": care_text
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"AI identify-care error: {type(e).__name__} - {str(e)}"
        )


@app.post("/compare-models")
async def compare_models(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Sadece JPG veya PNG desteklenir.")

    try:
        image_bytes = await file.read()
        result = compare_image_bytes(image_bytes)

        return {
            "message": "Model karşılaştırması tamamlandı",
            "result": result
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Model comparison error: {type(e).__name__} - {str(e)}"
        )