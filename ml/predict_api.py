import io
import torch
import torch.nn as nn

# PIL: görüntü dosyalarını açmak ve işlemek için kullanılan kütüphane
from PIL import Image

from pathlib import Path

# torchvision.models: hazır derin öğrenme modellerini içerir (ResNet vb.)
# transforms: görüntü üzerinde dönüşümler uygulamak için kullanılır
from torchvision import transforms, models

# burada çiçek id'sini isimlere çevirmek için kullanılan sözlüğü import ettim
from ml.flower_names import FLOWER_NAMES


# burada eğitilmiş model dosyasının yolunu belirledim
MODEL_PATH = Path(__file__).resolve().parent / "models" / "plant_transfer_finetuned_best.pth"

# burada model güven eşiğini belirledim
# model tahmin güveni bu değerden düşükse "unknown" dönecek
UNKNOWN_THRESHOLD = 0.45


# burada model ve sınıf isimlerini global değişken olarak tanımladım
_model = None
_class_names = None


def build_model(num_classes: int):

    # burada ResNet18 model mimarisini oluşturdum
    # weights=None diyerek hazır ağırlık kullanmadan model yapısını kurdum
    model = models.resnet18(weights=None)

    # burada resnet modelinin son katmanının giriş boyutunu aldım
    in_features = model.fc.in_features

    # burada son sınıflandırma katmanını kendi sınıf sayımıza göre değiştirdim
    model.fc = nn.Sequential(
        nn.Dropout(0.30),              # burada overfitting azaltmak için dropout kullandım
        nn.Linear(in_features, num_classes)  # burada sınıf sayısı kadar çıktı ürettim
    )

    return model


# Transform: modele gönderilmeden önce görüntüye uygulanan işlemler
_transform = transforms.Compose([

    # burada görüntüyü modelin beklediği boyuta getirdim
    transforms.Resize((224, 224)),

    # Tensor: PyTorch'un kullandığı sayısal veri formatıdır
    transforms.ToTensor(),

    # Normalize: görüntü değerlerini belirli aralığa ölçeklemek için kullanılır
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def load_model():

    # burada global değişkenleri kullanacağımı belirttim
    global _model, _class_names

    # burada model daha önce yüklenmişse tekrar yüklememek için kontrol yaptım
    if _model is not None:
        return _model, _class_names

    # burada eğitilmiş model dosyasını yükledim
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")

    # burada modelin eğitim sırasında kaydedilen sınıf isimlerini aldım
    _class_names = checkpoint["class_names"]

    # burada model mimarisini tekrar oluşturdum
    model = build_model(len(_class_names))

    # burada eğitilmiş ağırlıkları modele yükledim
    model.load_state_dict(checkpoint["model_state_dict"])

    # burada modeli evaluation moduna aldım
    # evaluation mode: modelin tahmin modunda çalışmasını sağlar
    model.eval()

    # burada modeli global değişkene kaydettim
    _model = model

    return _model, _class_names


def predict_image_bytes(image_bytes: bytes):

    # burada modeli yükledim
    model, class_names = load_model()

    # burada gelen byte verisini görüntüye dönüştürdüm
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # burada görüntüyü transform işleminden geçirip modele hazır hale getirdim
    image = _transform(image).unsqueeze(0)

    # unsqueeze: tensor boyutuna batch dimension eklemek için kullanılır
    # model tek görüntüyü bile batch formatında bekler

    with torch.no_grad():

        # burada görüntüyü modele gönderip tahmin çıktısını aldım
        outputs = model(image)

        # Softmax: model çıktısını olasılık değerlerine çevirir
        probs = torch.softmax(outputs, dim=1)[0]

    # burada en yüksek olasılığa sahip sınıfı buldum
    best_conf, best_idx = torch.max(probs, dim=0)

    best_conf = float(best_conf.item())
    best_idx = int(best_idx.item())

    # burada sınıf id'sini klasör isminden çıkardım
    class_id = int(class_names[best_idx].split("_")[1]) +1 

    # burada sınıf id'sini çiçek isimlerine çevirdim
    eng, tr = FLOWER_NAMES.get(class_id, ("unknown", "bilinmeyen"))

    # burada ingilizce ve türkçe isimleri birleştirdim
    plant_name = f"{eng} ({tr})"

    # burada model güveni düşükse unknown döndürdüm
    if best_conf < UNKNOWN_THRESHOLD:
        return {
            "plant": "unknown",
            "model": "ResNet18 Fine-Tuned"
        }

    # burada tahmin edilen bitki adını döndürdüm
    return {
        "plant": plant_name,
        "model": "ResNet18 Fine-Tuned"
    }