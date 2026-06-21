# NasoMind — Aplikasi Skrining Depresi (model biner XGBoost)

Prototipe aplikasi (peran: Aplikasi) untuk lomba esai statistika nasional.
Streamlit + model XGBoost biner dari tim Sains Data.

## Isi folder (untuk di-upload ke GitHub)
- app.py — aplikasi utama
- xgb_depression_model_Final.pkl — model biner (dari tim Sains Data)
- stable_features_Final.json — 20 fitur metabolit yang dipakai model
- metrik_model.json — diisi angka evaluasi (akurasi/F1) dari tim Sains Data
- contoh_data_pasien.csv — data contoh untuk diunggah (ada kolom berlebih untuk uji seleksi)
- artikel.md — artikel edukasi, bisa diedit seperti blog (tanpa Python)
- requirements.txt — daftar paket (termasuk xgboost)
- .streamlit/config.toml — tema warna
- .streamlit/secrets.toml.example — contoh API key & login Google
- .gitignore — melindungi secrets & data

## Cara kerja
1. Tim Sains Data mengekspor model (.pkl) + daftar fitur (.json) dari Colab.
2. Kedua file diletakkan di folder aplikasi (server).
3. Rumah Sakit mengunggah CSV. Kolom berlebih DIABAIKAN; aplikasi memilih
   variabel sesuai stable_features_Final.json.
4. Model memprediksi: "Terindikasi depresi" / "Tidak terindikasi depresi" +
   persentase keyakinan.
5. Pasien menarik hasil lewat ID Antrian di Dashboard.
6. Dashboard menampilkan metabolit paling berpengaruh + interpretasi AI (Gemini).

## Menjalankan lokal (Anaconda)
    pip install -r requirements.txt
    streamlit run app.py

## Mengisi metrik model
Buka metrik_model.json, isi angka dari evaluasi tim Sains Data, contoh:
    {"akurasi_total": 0.87, "f1": 0.85,
     "akurasi_per_kelas": {"Tidak terindikasi depresi": 0.88,
                            "Terindikasi depresi": 0.85}}

## Chat AI & interpretasi metabolit -> Gemini (gratis)
Ambil API key di https://aistudio.google.com, lalu isi di Secrets:
    GEMINI_API_KEY = "AIza..."
Tanpa key, chat tetap jalan (mode demo) dan interpretasi metabolit tidak tampil.

## Login pasien dengan Google
Isi blok [auth] di Secrets (lihat .streamlit/secrets.toml.example) dengan Client
ID & secret dari Google Cloud Console (redirect URI = URL app + /oauth2callback).

## Publikasi (Streamlit Community Cloud)
1. Upload semua file ke repo GitHub public (tanpa secrets.toml asli).
2. share.streamlit.io -> Create app -> pilih repo + app.py -> Deploy.
3. Isi Secrets (GEMINI_API_KEY + blok [auth]) di Settings.
4. Dapat URL https://NAMAAPP.streamlit.app untuk disisipkan di esai.

Catatan: bila model gagal dimuat di Cloud, samakan versi xgboost & scikit-learn
di requirements.txt dengan versi yang dipakai tim Sains Data di Colab.

## Penting (integritas ilmiah)
Model ini BINER. Tampilan "Terindikasi/Tidak terindikasi" sudah sesuai sifat
model. Interpretasi metabolit oleh AI bersifat edukatif, bukan klaim medis, dan
hasil aplikasi adalah skrining — bukan diagnosis.
