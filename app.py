import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json
import plotly.express as px

# Konfigurasi halaman
st.set_page_config(page_title="iGutHealth", page_icon="🏥", layout="wide")

# Sedikit penyesuaian gaya supaya tampilan lebih bersih (tren 2026)
st.markdown(
    """
    <style>
      .stApp { background-color: #FFFFFF; }
      section[data-testid="stSidebar"] { background-color: #FFBFBF; }
      section[data-testid="stSidebar"] * { color: #FFFFFF; }
      div[data-testid="stMetricValue"] { color: #FFBFBF; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Konstanta ----------
NAMA_FILE_MODEL = "model_iguthealth.joblib"
FITUR = ["Diversitas_Mikroba", "Butirat", "Propionat", "Asetat",
         "Serotonin", "GABA", "Dopamin",
         "Corynebacterium", "Staphylococcus", "Moraxella",
         "Jam_Tidur", "Skor_Stres", "Usia"]
KELAS = ["Rendah", "Sedang", "Tinggi"]
WARNA_KELAS = {"Rendah": "#FFDE4D", "Sedang": "#FFB22C", "Tinggi": "#FF4C4C"}

TEKS_KRISIS = (
    "**Hotline Kesehatan Mental Bunuh Diri**\n"
    "**Kementerian Kesehatan**\n\n"
    "**119 ext. 8** jika merasa ingin mengakhiri hidup\n"
    "WhatsApp jika butuh cerita dengan seseorang\n"
)
DISCLAIMER = ("⚠️ iGutHealth merupakan aplikasi deteksi, bukan diagnosis."
              "Anda wajib konseling kembali dengan pihak rumah sakit.")

# Interpretasi tiap tingkat (ditampilkan ke pasien)
INTERPRETASI = {
    "Rendah": "Indikasi gejala depresi tergolong RINGAN. Tetap jaga gaya "
              "hidup sehat sebagai pencegahan.",
    "Sedang": "Indikasi gejala depresi tergolong SEDANG. Perlu perhatian, "
              "pemantauan, dan sebaiknya berkonsultasi dengan tenaga profesional.",
    "Tinggi": "Indikasi gejala depresi tergolong BERAT. Sangat disarankan segera "
              "berkonsultasi dengan psikiater/dokter.",
}

# Penyimpanan hasil ke file (mensimulasikan database backend). Dipakai agar
# pasien (sesi berbeda dari rumah sakit) tetap bisa menarik hasil lewat ID.
FILE_HASIL = "hasil_pasien.json"


def simpan_hasil(hasil):
    try:
        with open(FILE_HASIL, "w", encoding="utf-8") as f:
            json.dump(hasil, f)
    except Exception:
        pass


def muat_hasil():
    try:
        with open(FILE_HASIL, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


# Memuat / membangun model 
@st.cache_resource
def muat_model():
    """Memuat file model dari backend. Jika file belum ada (atau gagal dibaca),
    aplikasi membangun model sintetis agar tetap bisa didemokan.
    Saat file model asli dari tim Sains Data sudah tersedia, cukup letakkan
    sebagai 'model_iguthealth.joblib' di folder yang sama -> otomatis dipakai."""
    if os.path.exists(NAMA_FILE_MODEL):
        try:
            return joblib.load(NAMA_FILE_MODEL)
        except Exception:
            pass
    return bangun_model_sintetis()


def bangun_model_sintetis():
    """Versi simulasi backend (yang sesungguhnya dikerjakan tim di Colab)."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import recall_score, accuracy_score, f1_score, confusion_matrix

    rng = np.random.RandomState(42)
    n = 600
    X = pd.DataFrame(rng.rand(n, len(FITUR)), columns=FITUR)
    X["Jam_Tidur"] = np.round(3 + X["Jam_Tidur"] * 6, 1)
    X["Skor_Stres"] = np.round(X["Skor_Stres"] * 10, 1)
    X["Usia"] = (18 + X["Usia"] * 50).astype(int)
    skor = (-X["Diversitas_Mikroba"] - X["Butirat"] - X["Serotonin"]
            + X["Skor_Stres"] / 10 + (8 - X["Jam_Tidur"]) / 8)
    y = pd.qcut(skor, q=3, labels=KELAS)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25,
                                          random_state=42, stratify=y)
    pipe = Pipeline([("scaler", StandardScaler()),
                     ("rf", RandomForestClassifier(n_estimators=200, random_state=42))])
    pipe.fit(Xtr, ytr)
    yp = pipe.predict(Xte)
    rec = recall_score(yte, yp, labels=KELAS, average=None)
    return {
        "pipeline": pipe,
        "fitur": FITUR,
        "kelas": KELAS,
        "akurasi_per_kelas": {k: round(float(r), 3) for k, r in zip(KELAS, rec)},
        "akurasi_total": round(float(accuracy_score(yte, yp)), 3),
        "f1_macro": round(float(f1_score(yte, yp, average="macro")), 3),
        "confusion": confusion_matrix(yte, yp, labels=KELAS).tolist(),
    }


MODEL = muat_model()


# Skenario solusi berdasarkan tingkat MDD
def solusi_berdasarkan_tingkat(tingkat):
    """Mengembalikan (judul, daftar_rekomendasi, butuh_hotline).
    Aturan keputusan dibuat eksplisit agar transparan dan bisa diaudit."""
    if tingkat == "Rendah":
        rek = ["Pertahankan tidur 7-9 jam dengan jadwal teratur.",
               "Pola makan ramah gut-microbiome: tinggi serat, sayur, makanan "
               "fermentasi (yogurt, tempe), batasi makanan ultra-proses.",
               "Aktivitas fisik ringan-sedang 150 menit/minggu.",
               "Jaga koneksi sosial dan latihan relaksasi/mindfulness."]
        return ("Tingkat Rendah - Pencegahan & Gaya Hidup", rek, False)
    if tingkat == "Sedang":
        rek = ["Lakukan psikoedukasi terstruktur dan pemantauan gejala harian.",
               "Jadwalkan konsultasi dengan psikolog dalam waktu dekat.",
               "Pertimbangkan skrining lanjutan (mis. PHQ-9) bersama tenaga ahli.",
               "Perkuat dukungan gaya hidup (tidur, nutrisi, aktivitas fisik)."]
        return ("Tingkat Sedang - Intervensi Terpandu", rek, False)
    rek = ["SANGAT disarankan segera berkonsultasi dengan psikiater/dokter.",
           "Susun rencana keselamatan bersama keluarga/orang terpercaya.",
           "Jangan menunda penanganan; lakukan evaluasi klinis menyeluruh.",
           "Gunakan layanan darurat di bawah ini bila merasa tidak aman."]
    return ("Tingkat Tinggi - Rujukan Profesional Segera", rek, True)


# ---------- Status sesi (menyimpan data lintas interaksi) ----------
def init_state():
    default = {"peran": None, "df_hasil": None, "hasil": {},
               "id_aktif": None, "chat": [], "nama_pasien": None}
    for k, v in default.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


# ---------- Login Google (autentikasi OIDC bawaan Streamlit) ----------
def login_google_aktif():
    """True jika pengguna sudah login lewat Google. Aman dipanggil walau
    konfigurasi [auth] belum diisi (mengembalikan False)."""
    try:
        return bool(st.user.is_logged_in)
    except Exception:
        return False


def auth_dikonfigurasi():
    """Cek apakah blok [auth] sudah ada di secrets (login Google siap)."""
    try:
        return "auth" in st.secrets
    except Exception:
        return False


# Login seamless: kalau sudah login Google, langsung dianggap sebagai pasien.
if st.session_state.peran is None and login_google_aktif():
    st.session_state.peran = "pasien"
    try:
        st.session_state.nama_pasien = st.user.name
    except Exception:
        st.session_state.nama_pasien = None


# =========================================================
# HALAMAN LOGIN
# =========================================================
def halaman_login():
    kolom = st.columns([1, 1.4, 1])
    with kolom[1]:
        st.markdown("<h1 style='color:#0F766E;text-align:center'>iGutHealth</h1>",
                    unsafe_allow_html=True)
        st.markdown("<p style='text-align:center;color:#6B7280'>Deteksi dini "
                    "tingkat depresi berbasis mikrobioma</p>",
                    unsafe_allow_html=True)
        tab_rs, tab_pasien = st.tabs(["Rumah Sakit", "Pasien"])

        with tab_rs:
            st.text_input("Email instansi", key="rs_email",
                          placeholder="rs@contoh.id")
            st.text_input("Kata sandi", type="password", key="rs_pass",
                          placeholder="Masukkan kata sandi")
            st.write("")   # penyeimbang jarak agar rata dengan jarak email->sandi
            if st.button("Masuk sebagai Rumah Sakit", use_container_width=True):
                st.session_state.peran = "rumah_sakit"
                st.rerun()

        with tab_pasien:
            st.write("Masuk cukup dengan akun Google Anda, tanpa perlu mengingat "
                     "email/kata sandi.")
            if auth_dikonfigurasi():
                st.button("Lanjutkan dengan Google", use_container_width=True,
                          on_click=st.login)
            else:
                # Login Google aktif setelah [auth] diisi di Secrets.
                # Sementara (untuk uji coba lokal) sediakan masuk mode demo.
                st.caption("Login Google aktif setelah dikonfigurasi di Secrets. "
                           "Untuk uji coba, gunakan tombol di bawah.")
                if st.button("Masuk (mode demo)", use_container_width=True):
                    st.session_state.peran = "pasien"
                    st.rerun()


# =========================================================
# HALAMAN-HALAMAN
# =========================================================
def hal_tentang():
    st.header("Tentang Aplikasi")
    with st.container(border=True):
        st.write("**iGutHealth** merupakan aplikasi deteksi depresi berbasis data "
                 "metabolomik dan mikrobiota menggunakan model *machine learning* "
                 "untuk transformasi kesehatan Indonesia.")
    with st.container(border=True):
        st.markdown("**Tim penyusun:**")
        st.markdown("- Bioteknologi : ______________________\n"
                    "- Sains Data   : ______________________\n"
                    "- Aplikasi     : ______________________")
        st.caption("Versi prototipe untuk lomba esai statistika nasional.")


def hal_unggah():
    st.header("Unggah Data")
    with st.container(border=True):
        st.write("Unggah file CSV berisi data pasien.")
        st.caption("Kolom wajib: ID_Antrian + " + ", ".join(FITUR))
        berkas = st.file_uploader("Pilih file")
        if berkas is not None:
            try:
                df = pd.read_csv(berkas)
                kurang = [c for c in FITUR if c not in df.columns]
                if kurang:
                    st.error("Kolom belum ada: " + ", ".join(kurang))
                    return
                # Prediksi kelas + probabilitas tiap kategori
                X = df[FITUR]
                pred = MODEL["pipeline"].predict(X)
                prob = MODEL["pipeline"].predict_proba(X)
                kelas_model = list(MODEL["pipeline"].classes_)

                df = df.copy()
                df["Prediksi"] = pred
                hasil = {}
                for i in range(len(df)):
                    idq = str(df.iloc[i].get("ID_Antrian", "A-%03d" % (i + 1)))
                    p = {k: float(prob[i][kelas_model.index(k)]) for k in kelas_model}
                    hasil[idq] = {"prediksi": str(pred[i]), "prob": p}

                st.session_state.df_hasil = df
                st.session_state.hasil = hasil
                simpan_hasil(hasil)   # agar pasien (sesi lain) bisa menariknya
                st.success("Berhasil. %d data diproses oleh model. "
                           "Buka menu Dashboard." % len(df))
                st.dataframe(df[["ID_Antrian", "Prediksi"]]
                             if "ID_Antrian" in df.columns else df[["Prediksi"]],
                             use_container_width=True, height=240)
            except Exception as e:
                st.error("Gagal memproses: " + str(e))


def kartu_akurasi_kelas():
    with st.container(border=True):
        st.subheader("Akurasi Model per Kategori (validasi)")
        c1, c2 = st.columns(2)
        c1.metric("Akurasi total", "%.1f%%" % (MODEL["akurasi_total"] * 100))
        c2.metric("F1-macro", "%.2f" % MODEL["f1_macro"])
        ak = MODEL["akurasi_per_kelas"]
        d = pd.DataFrame({"Kategori": list(ak.keys()),
                          "Akurasi (%)": [v * 100 for v in ak.values()]})
        fig = px.bar(d, x="Kategori", y="Akurasi (%)", color="Kategori",
                     color_discrete_map=WARNA_KELAS, range_y=[0, 100])
        fig.update_layout(showlegend=False, height=320)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Catatan: angka ini = recall (sensitivitas) tiap kategori dari "
                   "data validasi -> metrik kualitas model, bukan kondisi pasien.")


def hal_dashboard():
    st.header("Dashboard")
    st.info(DISCLAIMER)   # disclaimer di bagian header dashboard

    if st.session_state.peran == "rumah_sakit":
        if st.session_state.df_hasil is None:
            st.warning("Belum ada data. Silakan unggah data dulu di menu Unggah Data.")
            return
        df = st.session_state.df_hasil
        with st.container(border=True):
            st.subheader("Deskripsi Data")
            c1, c2, c3 = st.columns(3)
            c1.metric("Jumlah pasien", df.shape[0])
            c2.metric("Jumlah kolom", df.shape[1])
            c3.metric("Sel kosong", int(df.isna().sum().sum()))
        with st.container(border=True):
            st.subheader("Distribusi Prediksi Tingkat MDD")
            hit = df["Prediksi"].value_counts().reindex(KELAS).fillna(0)
            d = pd.DataFrame({"Kategori": KELAS, "Jumlah": hit.values})
            fig = px.bar(d, x="Kategori", y="Jumlah", color="Kategori",
                         color_discrete_map=WARNA_KELAS)
            fig.update_layout(showlegend=False, height=320)
            st.plotly_chart(fig, use_container_width=True)
        kartu_akurasi_kelas()

    else:  # pasien
        # Box input ID antrian (menggantikan menu terpisah)
        with st.container(border=True):
            st.subheader("Cek Hasil Anda")
            idq = st.text_input("Masukkan ID Antrian", placeholder="contoh: A-001")
            if st.button("Lihat Hasil"):
                semua = muat_hasil()
                if idq.strip() in semua:
                    st.session_state.id_aktif = idq.strip()
                else:
                    st.session_state.id_aktif = None
                    st.warning("ID Antrian belum ada. Pastikan Rumah Sakit sudah "
                               "mengunggah data Anda.")

        semua = muat_hasil()
        if st.session_state.id_aktif and st.session_state.id_aktif in semua:
            _kartu_hasil_pasien(semua[st.session_state.id_aktif])


def _kartu_hasil_pasien(data):
    tingkat = data["prediksi"]
    prob = data["prob"]
    keyakinan = prob.get(tingkat, 0) * 100

    with st.container(border=True):
        st.subheader("Hasil untuk ID %s" % st.session_state.id_aktif)
        st.markdown("Tingkat depresi diprediksi: "
                    "<span style='color:%s;font-size:26px;font-weight:700'>%s</span>"
                    % (WARNA_KELAS.get(tingkat, "#0B1F1C"), tingkat),
                    unsafe_allow_html=True)
        st.write("Model memperkirakan tingkat depresi Anda **%s** dengan tingkat "
                 "keyakinan **%.0f%%**." % (tingkat, keyakinan))
        st.info("Interpretasi: " + INTERPRETASI.get(tingkat, ""))

    judul, rekomendasi, butuh_hotline = solusi_berdasarkan_tingkat(tingkat)
    with st.container(border=True):
        st.subheader("Rekomendasi: " + judul)
        for r in rekomendasi:
            st.markdown("- " + r)
        if butuh_hotline:
            st.error(TEKS_KRISIS)


def hal_riwayat():
    st.header("Riwayat Konsultasi")
    with st.container(border=True):
        # Data contoh; di produksi diambil dari basis data rumah sakit.
        riwayat = pd.DataFrame({
            "Tanggal": ["12 Mei 2026", "02 Apr 2026"],
            "Faskes": ["RS Diponegoro", "RS Diponegoro"],
            "Tingkat": ["Sedang", "Tinggi"],
            "Catatan": ["Kontrol rutin", "Rujukan psikiater"],
        })
        st.dataframe(riwayat, use_container_width=True, hide_index=True)


def muat_artikel():
    """Membaca artikel dari file artikel.md (bisa diedit seperti blog, tanpa
    menyentuh kode Python). Tiap artikel diawali baris '## Judul'."""
    if not os.path.exists("artikel.md"):
        return []
    teks = open("artikel.md", encoding="utf-8").read()
    daftar, judul, isi = [], None, []
    for baris in teks.splitlines():
        if baris.startswith("## "):
            if judul is not None:
                daftar.append((judul, "\n".join(isi).strip()))
            judul, isi = baris[3:].strip(), []
        elif judul is not None:
            isi.append(baris)
    if judul is not None:
        daftar.append((judul, "\n".join(isi).strip()))
    return daftar


def hal_artikel():
    st.header("Artikel Edukasi")
    daftar = muat_artikel()
    if not daftar:
        st.info("Belum ada artikel. Tambahkan tulisan di file artikel.md.")
        return
    for judul, isi in daftar:
        with st.container(border=True):
            st.subheader(judul)
            st.markdown(isi)   # mendukung teks tebal, daftar, tautan (seperti blog)


# ---------- Pengaturan platform Chat GenAI ----------
# Nama model bisa diganti sesuai kebutuhan/biaya.
MODEL_GEMINI = "gemini-2.5-flash"    # gratis di free tier Google AI Studio
MODEL_CLAUDE = "claude-sonnet-4-6"   # alternatif lebih murah: "claude-haiku-4-5-20251001"
MODEL_OPENAI = "gpt-4o-mini"         # alternatif: "gpt-4o"

# System prompt: mengatur peran & batas keamanan AI (sangat penting untuk topik ini)
SISTEM_PROMPT = (
    "Anda adalah pendamping edukasi kesehatan mental pada aplikasi iGutHealth. "
    "Anda BUKAN pengganti tenaga profesional: jangan mendiagnosis, jangan "
    "meresepkan obat, jangan menggantikan psikolog/psikiater/dokter. Berikan "
    "dukungan empatik dan psikoedukasi singkat seputar gaya hidup sehat (tidur, "
    "nutrisi ramah mikrobioma usus, aktivitas fisik, relaksasi, koneksi sosial). "
    "Untuk hal klinis, selalu sarankan konsultasi ke profesional. Jika pengguna "
    "menunjukkan tanda krisis atau ingin menyakiti diri, arahkan dengan lembut ke "
    "layanan 119 ekstensi 8 (SEJIWA) atau www.healing119.id, dan jangan memberi "
    "detail apa pun soal cara menyakiti diri. Jawab ringkas, hangat, Bahasa Indonesia."
)


def ada_kunci(nama):
    """Cek apakah API key tersedia di Streamlit Secrets."""
    try:
        return nama in st.secrets and bool(st.secrets[nama])
    except Exception:
        return False


def jawaban_dari_api(riwayat):
    """Memanggil API GenAI sungguhan. Prioritas: Gemini (Google, gratis) lalu
    Claude (Anthropic) lalu ChatGPT (OpenAI). API key diambil dari Streamlit
    Secrets (server), TIDAK ditulis di dalam kode. 'riwayat' = list (peran, teks)."""
    pesan_terakhir = riwayat[-12:]   # batasi konteks agar hemat token

    if ada_kunci("GEMINI_API_KEY"):
        from google import genai
        from google.genai import types
        klien = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
        # Gemini memakai peran "user" dan "model" (bukan "assistant")
        isi = []
        for r, t in pesan_terakhir:
            peran_g = "user" if r == "user" else "model"
            isi.append(types.Content(role=peran_g,
                                     parts=[types.Part.from_text(text=t)]))
        resp = klien.models.generate_content(
            model=MODEL_GEMINI, contents=isi,
            config=types.GenerateContentConfig(system_instruction=SISTEM_PROMPT,
                                               max_output_tokens=400))
        return resp.text

    if ada_kunci("ANTHROPIC_API_KEY"):
        import anthropic
        klien = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
        pesan = [{"role": r, "content": t} for r, t in pesan_terakhir]
        resp = klien.messages.create(model=MODEL_CLAUDE, max_tokens=400,
                                     system=SISTEM_PROMPT, messages=pesan)
        return resp.content[0].text

    if ada_kunci("OPENAI_API_KEY"):
        from openai import OpenAI
        klien = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        pesan = [{"role": "system", "content": SISTEM_PROMPT}]
        pesan += [{"role": r, "content": t} for r, t in pesan_terakhir]
        resp = klien.chat.completions.create(model=MODEL_OPENAI,
                                             max_tokens=400, messages=pesan)
        return resp.choices[0].message.content

    return None   # tidak ada API key -> pakai mode aturan di bawah


def jawaban_aturan(p):
    """Cadangan sederhana ketika API key belum dipasang (mode demo)."""
    if "tidur" in p:
        return ("Coba jadwal tidur teratur 7-9 jam dan kurangi layar sebelum tidur. "
                "Bila gangguan tidur menetap, konsultasikan ke dokter.")
    if "makan" in p or "diet" in p:
        return ("Perbanyak serat, sayur, dan makanan fermentasi untuk mendukung "
                "mikrobioma usus. Untuk rencana gizi spesifik, temui ahli gizi.")
    return ("Terima kasih. Untuk hal yang bersifat medis sebaiknya dikonsultasikan "
            "dengan psikolog/psikiater. Ada topik gaya hidup sehat yang ingin "
            "Anda tanyakan?")


def jawaban_ai(pesan, riwayat):
    # 1) Lapisan keamanan: deteksi krisis secara pasti, tidak bergantung pada API
    p = pesan.lower()
    krisis = ["bunuh diri", "mengakhiri", "menyakiti diri", "putus asa",
              "tidak ada harapan", "mengakhiri hidup"]
    if any(k in p for k in krisis):
        return ("Saya khawatir dengan apa yang Anda rasakan, dan Anda tidak "
                "sendirian. Mohon segera hubungi **119 ekstensi 8 (SEJIWA)** atau "
                "buka **www.healing119.id** untuk bicara dengan konselor.")
    # 2) Coba panggil API GenAI sungguhan
    try:
        balas = jawaban_dari_api(riwayat)
        if balas:
            return balas
    except Exception:
        return ("(Mode demo) Koneksi ke layanan AI sedang tidak tersedia. "
                "Untuk hal medis, silakan konsultasi ke psikolog/psikiater.")
    # 3) Cadangan berbasis aturan (jika API key belum dipasang)
    return jawaban_aturan(p)


def hal_chat():
    st.header("Chat AI (Pendamping Edukasi)")
    st.caption("Pendamping edukasi, bukan pengganti tenaga profesional.")
    # Tampilkan status koneksi platform GenAI
    if ada_kunci("GEMINI_API_KEY"):
        st.caption("🟢 Terhubung ke Gemini (Google).")
    elif ada_kunci("ANTHROPIC_API_KEY"):
        st.caption("🟢 Terhubung ke Claude (Anthropic).")
    elif ada_kunci("OPENAI_API_KEY"):
        st.caption("🟢 Terhubung ke ChatGPT (OpenAI).")
    else:
        st.caption("🟡 Mode demo (tanpa API). Tambahkan API key di Streamlit "
                   "Secrets untuk mengaktifkan AI sungguhan.")

    for peran, isi in st.session_state.chat:
        with st.chat_message(peran):
            st.markdown(isi)
    pesan = st.chat_input("Tulis pesan...")
    if pesan:
        st.session_state.chat.append(("user", pesan))
        balas = jawaban_ai(pesan, st.session_state.chat)
        st.session_state.chat.append(("assistant", balas))
        st.rerun()


def hal_callcenter():
    st.header("Call Center Depresi")
    with st.container(border=True):
        st.subheader("Anda tidak sendirian.")
        st.write("Jika kamu merasa butuh bantuan atau cerita dengan seseorang, "
                 "klik dan hubungi hotline ini.")
        st.markdown(TEKS_KRISIS)


# =========================================================
# ROUTING UTAMA
# =========================================================
if st.session_state.peran is None:
    halaman_login()
    st.stop()

# Sidebar: identitas + menu sesuai peran
st.sidebar.title("iGutHealth")
if st.session_state.peran == "rumah_sakit":
    st.sidebar.caption("Masuk sebagai: Rumah Sakit")
else:
    nama = st.session_state.get("nama_pasien")
    st.sidebar.caption("Masuk sebagai: Pasien" + (" (%s)" % nama if nama else ""))

if st.session_state.peran == "rumah_sakit":
    menu = {"Tentang Aplikasi": hal_tentang,
            "Unggah Data": hal_unggah, "Dashboard": hal_dashboard}
else:
    menu = {"Tentang Aplikasi": hal_tentang, "Dashboard": hal_dashboard,
            "Riwayat Konsultasi": hal_riwayat, "Artikel Edukasi": hal_artikel,
            "Chat AI": hal_chat, "Call Center": hal_callcenter}

pilihan = st.sidebar.radio("Menu", list(menu.keys()))
st.sidebar.divider()
if st.sidebar.button("Keluar"):
    pakai_google = login_google_aktif()
    # Bersihkan status sesi aplikasi
    for k in ["peran", "df_hasil", "hasil", "id_aktif", "chat", "nama_pasien"]:
        st.session_state[k] = ([] if k == "chat" else
                               ({} if k == "hasil" else None))
    if pakai_google:
        st.logout()      # hapus sesi Google + cookie, lalu rerun otomatis
    else:
        st.rerun()

# Jalankan fungsi halaman terpilih
menu[pilihan]()
