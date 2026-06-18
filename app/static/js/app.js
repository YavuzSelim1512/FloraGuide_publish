const btn = document.getElementById("btn");
const fileInput = document.getElementById("fileInput");
const statusText = document.getElementById("status");

const result = document.getElementById("result");
const plant = document.getElementById("plant");
const model = document.getElementById("model");
const careText = document.getElementById("careText");

const previewCard = document.getElementById("previewCard");
const previewImage = document.getElementById("previewImage");

function showError(message) {
  statusText.innerText = message;
  statusText.className = "status error";
}

function showSuccess(message) {
  statusText.innerText = message;
  statusText.className = "status success";
}

fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];

  if (!file) {
    previewCard.classList.add("hidden");
    return;
  }

  const imageUrl = URL.createObjectURL(file);
  previewImage.src = imageUrl;
  previewCard.classList.remove("hidden");
});

btn.onclick = async () => {
  const file = fileInput.files[0];

  if (!file) {
    showError("Lütfen önce bir fotoğraf seç.");
    return;
  }

  showSuccess("Fotoğraf analiz ediliyor...");
  result.classList.add("hidden");

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch("/identify-care", {
      method: "POST",
      body: formData
    });

    const data = await res.json();

    if (!res.ok) {
      showError(data.detail || "Bir hata oluştu.");
      return;
    }

    plant.innerText = data.plant || "-";
    model.innerText = data.model || "-";
    careText.innerText = data.care_recommendation || "Bakım önerisi bulunamadı.";

    result.classList.remove("hidden");
    showSuccess("Analiz tamamlandı.");
  } catch (error) {
    showError("Sunucuya bağlanırken hata oluştu.");
  }
};
