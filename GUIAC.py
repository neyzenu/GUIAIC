import os
import tkinter as tk
from tkinter import filedialog, Label, Button, Entry, StringVar
from PIL import Image, ImageTk
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.models import resnet18

# ROCm desteğini kontrol etme ve ROCm aygıtını ayarlama
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Küresel sınıf isimleri listesi
class_names = []


# Basit bir özel veri kümesi sınıfı
class CustomDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        image_path, label = self.file_paths[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# Model ve eğitim fonksiyonu
def create_and_train_model(train_loader, val_loader, num_classes, num_epochs, progress_callback):
    model = resnet18()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    total_batches = num_epochs * len(train_loader)  # Toplam batch sayısı hesaplanır
    for epoch in range(num_epochs):
        # Eğitim aşaması
        total_train_loss, total_correct = 0.0, 0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            total_correct += (outputs.argmax(1) == labels).sum().item()

            # Batch başına yüzde ilerleme hesabı
            current_batch = epoch * len(train_loader) + (i + 1)
            percent_complete = (current_batch / total_batches) * 100
            progress_callback(epoch, num_epochs, train_loss=total_train_loss / (i + 1),
                              val_loss=0, train_acc=0, val_acc=0, percent_complete=percent_complete)

        train_loss = total_train_loss / len(train_loader)
        train_accuracy = total_correct / len(train_loader.dataset)

        # Doğrulama aşaması
        model.eval()
        total_val_loss, total_correct = 0.0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()
                total_correct += (outputs.argmax(1) == labels).sum().item()

        val_loss = total_val_loss / len(val_loader)
        val_accuracy = total_correct / len(val_loader.dataset)

        # Epoch sonunda doğrulama sonuçlarını güncelle
        progress_callback(epoch, num_epochs, train_loss=train_loss, val_loss=val_loss,
                          train_acc=train_accuracy, val_acc=val_accuracy, percent_complete=percent_complete)
        model.train()  # Modeli yeniden eğitim moduna al

    return model


# Sınıf isimlerini dosyaya yazma
def save_class_names(classes):
    with open('class_names.txt', 'w') as f:
        for class_name in classes:
            f.write(f"{class_name}\n")


# Sınıf isimlerini dosyadan okuma
def load_class_names():
    with open('class_names.txt', 'r') as f:
        return [line.strip() for line in f]


# Tahmin için kullanılan dönüşüm
data_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


# Tahmin fonksiyonu
def predict_image(model, image_path):
    image = Image.open(image_path).convert('RGB')
    image = data_transforms(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return class_names[predicted.item()]


# Tkinter Arayüz Sınıfı
class AIApp:
    def __init__(self, master):
        self.root = master
        self.root.title("AI Model Eğitim ve Sınıflandırma")
        self.root.geometry("500x750")

        # Sınıf sayısını belirleme alanı
        self.class_label = Label(master, text="Kaç sınıf var?", font=("Arial", 12))
        self.class_label.pack(pady=5)

        self.num_classes_var = StringVar()
        self.num_classes_entry = Entry(master, textvariable=self.num_classes_var)
        self.num_classes_entry.pack(pady=5)

        self.set_classes_button = Button(master, text="Sınıf Sayısını Ayarla", command=self.ask_for_classes)
        self.set_classes_button.pack(pady=10)

        # Epoch sayısı için giriş alanı
        self.epoch_label = Label(master, text="Epoch Sayısı:", font=("Arial", 12))
        self.epoch_label.pack(pady=5)

        self.num_epochs_var = StringVar(value="5")  # Varsayılan epoch sayısı 5 olarak ayarlandı
        self.num_epochs_entry = Entry(master, textvariable=self.num_epochs_var)
        self.num_epochs_entry.pack(pady=5)

        # Eğitim düğmesi
        self.train_button = Button(master, text="Eğitimi Başlat", command=self.start_training_thread, state=tk.DISABLED)
        self.train_button.pack(pady=10)

        # Fotoğraf yükleme ve tahmin düğmeleri
        self.load_button = Button(master, text="Fotoğraf Yükle", command=self.load_image)
        self.load_button.pack(pady=10)
        self.predict_button = Button(master, text="Tahmin Et", command=self.predict, state=tk.DISABLED)
        self.predict_button.pack(pady=10)

        # Resim ve sonuç gösterimi
        self.image_label = Label(master, text="Fotoğraf Yükleyin", width=40, height=20, borderwidth=2, relief="solid")
        self.image_label.pack(pady=10)
        self.result_label = Label(master, text="Tahmin edilen sınıf: -", font=("Arial", 14))
        self.result_label.pack(pady=10)

        # Eğitim ve doğrulama sonuçları
        self.progress_label = Label(master, text="Durum: Bekleniyor", font=("Arial", 10), fg="blue")
        self.progress_label.pack(pady=5)
        self.train_loss_label = Label(master, text="Eğitim Kaybı: -", font=("Arial", 10), fg="black")
        self.train_loss_label.pack(pady=5)
        self.val_loss_label = Label(master, text="Doğrulama Kaybı: -", font=("Arial", 10), fg="black")
        self.val_loss_label.pack(pady=5)
        self.train_acc_label = Label(master, text="Eğitim Doğruluğu: -", font=("Arial", 10), fg="black")
        self.train_acc_label.pack(pady=5)
        self.val_acc_label = Label(master, text="Doğrulama Doğruluğu: -", font=("Arial", 10), fg="black")
        self.val_acc_label.pack(pady=5)
        self.percent_label = Label(master, text="Eğitim İlerlemesi: %0.0", font=("Arial", 10), fg="green")
        self.percent_label.pack(pady=5)

        # Yüklenecek fotoğrafın yolu
        self.image_path = None
        self.dataset_paths = []

        # Eğitilmiş bir modeli yüklemeyi dener
        self.model = None
        self.load_trained_model()

    # Önceden eğitilmiş modeli yükleme
    def load_trained_model(self):
        if os.path.exists("trained_model.pth") and os.path.exists("class_names.txt"):
            try:
                global class_names
                class_names = load_class_names()

                self.model = resnet18()
                self.model.fc = nn.Linear(self.model.fc.in_features, len(class_names))
                self.model.load_state_dict(torch.load("trained_model.pth", map_location=device))
                self.model = self.model.to(device)
                self.model.eval()

                self.progress_label.config(text="Önceden eğitilmiş model yüklendi.")
                self.predict_button.config(state=tk.NORMAL)  # Tahmin düğmesini etkinleştir
            except Exception as e:
                self.progress_label.config(text=f"Model yüklenemedi: {e}")

    # Eğitim için gerekli dosyaları ayarlama
    def ask_for_classes(self):
        try:
            num_classes = int(self.num_classes_var.get())
        except ValueError:
            self.progress_label.config(text="Lütfen geçerli bir sayı girin.")
            return

        # Sınıf dizinlerini seçme
        for _ in range(num_classes):
            folder_path = filedialog.askdirectory()
            if folder_path:
                class_name = os.path.basename(folder_path)
                class_names.append(class_name)
                self.dataset_paths.append((folder_path, len(class_names) - 1))

        save_class_names(class_names)
        self.train_button.config(state=tk.NORMAL)

    # Eğitim işlemini başlatmak için iş parçacığı
    def start_training_thread(self):
        import threading
        threading.Thread(target=self.start_training).start()

    # Eğitim sürecini başlatma
    def start_training(self):
        try:
            num_epochs = int(self.num_epochs_var.get())
        except ValueError:
            self.progress_label.config(text="Lütfen geçerli bir epoch sayısı girin.")
            return

        image_paths = []
        for folder_path, label in self.dataset_paths:
            for file_name in os.listdir(folder_path):
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append((os.path.join(folder_path, file_name), label))

        # Eğitim ve doğrulama veri kümesi ayırma
        transform = data_transforms
        dataset = CustomDataset(image_paths, transform=transform)
        val_size = int(0.2 * len(dataset))  # %20 doğrulama verisi
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        # Model eğitimi
        self.progress_label.config(text="Eğitim Başlatıldı")
        self.model = create_and_train_model(train_loader, val_loader, len(class_names), num_epochs,
                                            self.update_progress)
        torch.save(self.model.state_dict(), 'trained_model.pth')
        self.progress_label.config(text="Eğitim tamamlandı ve model kaydedildi.")
        self.predict_button.config(state=tk.NORMAL)  # Tahmin düğmesini etkinleştir

    # Eğitim ve doğrulama ilerlemesini güncelleme
    def update_progress(self, epoch, num_epochs, train_loss, val_loss, train_acc, val_acc, percent_complete):
        self.progress_label.config(text=f"Epoch {epoch + 1}/{num_epochs}")
        self.train_loss_label.config(text=f"Eğitim Kaybı: {train_loss:.4f}")
        self.val_loss_label.config(text=f"Doğrulama Kaybı: {val_loss:.4f}")
        self.train_acc_label.config(text=f"Eğitim Doğruluğu: %{train_acc * 100:.2f}")
        self.val_acc_label.config(text=f"Doğrulama Doğruluğu: %{val_acc * 100:.2f}")
        self.percent_label.config(text=f"Eğitim İlerlemesi: %{percent_complete:.2f}")
        self.root.update_idletasks()

    # Fotoğraf yükleme fonksiyonu
    def load_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if self.image_path:
            image = Image.open(self.image_path)
            image = image.resize((256, 256), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo

    # Tahmin fonksiyonu
    def predict(self):
        if self.image_path and self.model:
            predicted_class = predict_image(self.model, self.image_path)
            self.result_label.config(text=f"Tahmin edilen sınıf: {predicted_class}")
        else:
            self.result_label.config(text="Lütfen önce modeli eğitin veya yükleyin.")


# Uygulama başlatma
root = tk.Tk()
app = AIApp(root)
root.mainloop()
