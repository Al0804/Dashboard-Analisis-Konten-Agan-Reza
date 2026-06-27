import streamlit as st
import pandas as pd
import torch
import numpy as np
import re
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
import gspread
from google.oauth2.service_account import Credentials
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# ==============================================================================
# 1. KONFIGURASI HALAMAN & DATABASE GOOGLE SHEETS
# ==============================================================================
st.set_page_config(page_title="Dashboard Agan Reza", layout="wide")
st.title("Dashboard Analisis Komentar Review Episode")
st.write("Pantau mayoritas topik pembicaraan penonton secara Real-Time via Cloud Database.")

sembunyikan_menu = """
<style>
#MainMenu {visibility: hidden;}
[data-testid="stToolbar"] {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(sembunyikan_menu, unsafe_allow_html=True)

kolom_kategori = ['Diskusi_Cerita', 'Evaluasi_Teknis', 'Q&A', 'Permintaan_Konten', 'Apresiasi_Kreator']
kolom_visualisasi = kolom_kategori + ['Outlier']

sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 10})

@st.cache_resource
def koneksi_gsheet():
    try:
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
        creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=scopes)
        client = gspread.authorize(creds)
        sheet = client.open_by_url(st.secrets["gsheet_url"]).sheet1
        return sheet
    except Exception as e:
        return None

sheet = koneksi_gsheet()

if sheet is None:
    st.error("⚠️ Google Sheets API Belum Terhubung atau Secrets belum disetting!")
    with st.expander("Klik di sini untuk panduan setting rahasia Streamlit Cloud"):
        st.markdown("""
        **Cara hubungin ke Google Sheets:**
        1. Bikin Service Account di Google Cloud Console, download file JSON-nya.
        2. Bikin Google Sheets baru, terus **Share/Bagikan** email service account lu ke dalam sheet sebagai **Editor**.
        3. Di menu **Advanced Settings -> Secrets** (Streamlit Cloud), isi dengan format ini:
        ```toml
        gsheet_url = "PASTE_LINK_URL_GOOGLE_SHEETS_LU_DISINI"

        [gcp_service_account]
        type = "service_account"
        project_id = "..."
        private_key_id = "..."
        private_key = "..."
        client_email = "..."
        # ... isi sisanya persis sesuai isi file JSON yang lu download tadi!
        ```
        """)
    st.stop()

# ==============================================================================
# UTILITY FUNGSI GSHEETS
# ==============================================================================
@st.cache_data(ttl=60, show_spinner=False)
def baca_gsheet():
    if sheet is None: return pd.DataFrame()
    records = sheet.get_all_records()
    if not records:
        return pd.DataFrame(columns=['Series', 'Episode', 'Komentar_Teks'] + kolom_visualisasi)
    return pd.DataFrame(records)

def simpan_ke_gsheet(df_baru):
    if sheet is None: return
    data_list = df_baru.fillna("").values.tolist()
    if len(sheet.get_all_values()) == 0:
        sheet.append_row(df_baru.columns.tolist())
    sheet.append_rows(data_list)
    baca_gsheet.clear()  # data baru ditulis, paksa baca_gsheet ambil data segar lagi

def tulis_ulang_gsheet(df_total):
    if sheet is None: return
    sheet.clear()
    sheet.append_row(df_total.columns.tolist())
    if not df_total.empty:
        sheet.append_rows(df_total.fillna("").values.tolist())
    baca_gsheet.clear()  # data ditulis ulang, paksa baca_gsheet ambil data segar lagi

# --- INISIALISASI MEMORI STREAMLIT ---
if 'analisis_selesai' not in st.session_state:
    st.session_state.analisis_selesai = False
if 'df_final' not in st.session_state:
    st.session_state.df_final = None
if 'total_kategori' not in st.session_state:
    st.session_state.total_kategori = None
if 'kolom_teks_aktif' not in st.session_state:
    st.session_state.kolom_teks_aktif = 'komentar'
if 'save_status' not in st.session_state:
    st.session_state.save_status = None
if 'save_message' not in st.session_state:
    st.session_state.save_message = None
if 'crud_notif' not in st.session_state:
    st.session_state.crud_notif = None

# ==============================================================================
# 2. CACHE MODEL & FUNGSI YOUTUBE SCRAPING
# ==============================================================================
@st.cache_resource
def load_model():
    path_model = "./model_final_siap_pakai"
    tokenizer = AutoTokenizer.from_pretrained("indolem/indobertweet-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(path_model)
    return tokenizer, model

tokenizer, model = load_model()

def ekstrak_video_id(url):
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
    if match:
        return match.group(1)
    return None

def tarik_komentar_youtube(video_id, api_key, max_komen=2000):
    youtube = build('youtube', 'v3', developerKey=api_key)
    daftar_komentar = []
    try:
        request = youtube.commentThreads().list(
            part="snippet", videoId=video_id, maxResults=100, textFormat="plainText"
        )
        while request and len(daftar_komentar) < max_komen:
            response = request.execute()
            for item in response['items']:
                komen = item['snippet']['topLevelComment']['snippet']['textDisplay']
                daftar_komentar.append(komen)
                if len(daftar_komentar) >= max_komen:
                    break
            request = youtube.commentThreads().list_next(request, response)
        return daftar_komentar
    except HttpError as e:
        st.error(f"Terjadi kesalahan pada API YouTube. Pastikan API Key valid. Detail: {e}")
        return []

# ==============================================================================
# 3. SISTEM TAB NAVIGASI
# ==============================================================================
tab_analisis, tab_riwayat = st.tabs(["🔍 Analisis AI Real-Time", "🗂️ Riwayat & Tren Database"])

# ------------------------------------------------------------------------------
# TAB 1: MESIN ANALISIS
# ------------------------------------------------------------------------------
with tab_analisis:
    mode_input = st.radio(
        "Pilih Metode Analisis:",
        ('Tarik dari Link YouTube (Real-Time)', 'Unggah File CSV (Data Lama)')
    )

    df_mentah = None
    kolom_teks = 'komentar'
    mulai_proses = False

    if mode_input == 'Tarik dari Link YouTube (Real-Time)':
        st.markdown("### Analisis Video Terbaru")
        api_key = st.text_input("Masukkan YouTube API Key Anda:", type="password")
        url_video = st.text_input("Masukkan Link Video YouTube:")

        if st.button("Tarik & Analisis Komentar"):
            if not api_key or not url_video:
                st.warning("API Key dan Link Video wajib diisi!")
            else:
                vid_id = ekstrak_video_id(url_video)
                if vid_id:
                    # ✅ FIX #1: Reset state SEBELUM analisis baru dimulai
                    # agar form lama tidak ikut muncul selama proses berlangsung
                    st.session_state.analisis_selesai = False
                    with st.spinner('Menyedot komentar dari YouTube...'):
                        hasil_scraping = tarik_komentar_youtube(vid_id, api_key, max_komen=2000)
                    if hasil_scraping:
                        st.success(f"Berhasil menarik {len(hasil_scraping)} komentar!")
                        df_mentah = pd.DataFrame(hasil_scraping, columns=['komentar'])
                        mulai_proses = True
                else:
                    st.error("Link YouTube tidak valid. Gagal menemukan Video ID.")

    elif mode_input == 'Unggah File CSV (Data Lama)':
        st.markdown("### Masukkan Data Komentar")
        file_unggahan = st.file_uploader("Unggah file CSV komentar:", type=["csv"])
        if file_unggahan is not None:
            df_mentah = pd.read_csv(file_unggahan)
            kolom_teks = 'komentar_bersih' if 'komentar_bersih' in df_mentah.columns else 'komentar'
            if st.button("Mulai Analisis CSV"):
                # ✅ FIX #1 (CSV): Reset state sebelum analisis baru
                st.session_state.analisis_selesai = False
                mulai_proses = True

    # ==========================================================================
    # PROSES PREDIKSI UTAMA
    # ==========================================================================
    if mulai_proses and df_mentah is not None:
        teks_list = df_mentah[kolom_teks].fillna("").astype(str).tolist()
        total_data = len(teks_list)

        progress_bar = st.progress(0)
        status_teks = st.empty()
        semua_tebakan = []
        batch_size = 16

        with st.spinner('Model sedang menganalisis kalimat dengan cermat...'):
            for i in range(0, total_data, batch_size):
                batch_teks = teks_list[i : i + batch_size]

                max_len = 256
                batas_tengah = int((max_len - 2) / 2)

                tokens = tokenizer(batch_teks, truncation=False, padding=False)
                input_ids_list = []
                attention_mask_list = []

                for ids in tokens['input_ids']:
                    if len(ids) > max_len:
                        kepala = ids[1 : batas_tengah + 1]
                        ekor = ids[-(batas_tengah + 1) : -1]
                        ids_baru = [ids[0]] + kepala + ekor + [ids[-1]]
                        mask_baru = [1] * max_len
                    else:
                        ids_baru = ids
                        mask_baru = [1] * len(ids)

                    input_ids_list.append(ids_baru)
                    attention_mask_list.append(mask_baru)

                inputs = tokenizer.pad(
                    {'input_ids': input_ids_list, 'attention_mask': attention_mask_list},
                    padding='max_length',
                    max_length=max_len,
                    return_tensors="pt"
                )

                with torch.no_grad():
                    outputs = model(**inputs)
                probabilitas = torch.sigmoid(outputs.logits).numpy()
                tebakan = (probabilitas > 0.5).astype(int)
                semua_tebakan.extend(tebakan)

                progress_bar.progress(min((i + batch_size) / total_data, 1.0))
                status_teks.text(f"Menganalisis {min(i + batch_size, total_data)} / {total_data} komentar...")

            kolom_yang_didrop = [kol for kol in kolom_kategori if kol in df_mentah.columns]
            df_bersih = df_mentah.drop(columns=kolom_yang_didrop).reset_index(drop=True)
            df_hasil = pd.DataFrame(semua_tebakan, columns=kolom_kategori)
            df_hasil['Outlier'] = df_hasil.sum(axis=1).apply(lambda total: 1 if total == 0 else 0)

            st.session_state.df_final = pd.concat([df_bersih, df_hasil], axis=1)
            st.session_state.total_kategori = df_hasil[kolom_visualisasi].sum()
            st.session_state.kolom_teks_aktif = kolom_teks
            st.session_state.analisis_selesai = True

        st.rerun()

    # ==========================================================================
    # TAMPILKAN VISUALISASI DARI MEMORI
    # ✅ FIX #2 (KUNCI UTAMA): Ganti 'if' jadi 'elif'
    # Blok ini HANYA jalan kalau blok prediksi di atas TIDAK sedang jalan.
    # Ini yang mencegah form muncul dobel saat model lagi proses.
    # ==========================================================================
    elif st.session_state.analisis_selesai:
        st.success("Analisis AI Selesai dan Tersimpan di Memori Sementara!")

        st.markdown("### Ringkasan Topik Pembicaraan")
        st.bar_chart(st.session_state.total_kategori.sort_values(ascending=False))

        st.markdown("### Detail Hasil Pemisahan Komentar")
        st.dataframe(st.session_state.df_final[[st.session_state.kolom_teks_aktif] + kolom_visualisasi].head(100))

        st.markdown("---")
        st.markdown("### 💾 Simpan Hasil ke Cloud Database")
        st.info("Data disalin langsung ke Google Sheets Agan Reza secara Real-Time.")

        if st.session_state.save_status == 'success':
            st.success(st.session_state.save_message)
            st.session_state.save_status = None
            st.session_state.save_message = None
        elif st.session_state.save_status == 'error':
            st.error(st.session_state.save_message)
            st.session_state.save_status = None
            st.session_state.save_message = None

        df_temp = baca_gsheet()
        daftar_series_tersimpan = []
        if not df_temp.empty and 'Series' in df_temp.columns:
            daftar_series_tersimpan = sorted(df_temp['Series'].dropna().unique().tolist())
        opsi_dropdown = ["-- Pilih Series --"] + daftar_series_tersimpan + ["➕ Tambah Series Baru..."]

        with st.form(key='form_simpan', clear_on_submit=False):
            col1, col2 = st.columns(2)
            with col1:
                pilih_series = st.selectbox("Pilih Judul Series:", opsi_dropdown)
                input_rider_baru = st.text_input(
                    "Jika tambah baru, ketik nama Rider-nya saja (Contoh: Gavv, Geats):",
                    placeholder="Kosongkan jika sudah memilih dari list di atas."
                )
            with col2:
                input_episode = st.text_input(
                    "Nomor Episode (Cukup ketik angkanya saja, misal: 1, 12, 45):",
                    placeholder="Hanya Angka"
                )

            tombol_simpan = st.form_submit_button("Simpan ke Cloud Database")

            if tombol_simpan:
                is_valid = True
                final_series = ""

                if pilih_series == "-- Pilih Series --":
                    st.error("Silakan pilih atau tambah Series terlebih dahulu!")
                    is_valid = False
                elif pilih_series == "➕ Tambah Series Baru...":
                    if input_rider_baru.strip():
                        final_series = f"Kamen Rider {input_rider_baru.strip()}"
                    else:
                        st.error("Nama Rider baru belum diisi!")
                        is_valid = False
                else:
                    final_series = pilih_series

                final_episode = ""
                if input_episode.strip():
                    final_episode = f"Episode {input_episode.strip()}"
                else:
                    if is_valid:
                        st.error("Nomor episode wajib diisi!")
                    is_valid = False

                if is_valid:
                    with st.spinner(f'Menyimpan data "{final_series} - {final_episode}" ke Cloud Database...'):
                        df_simpan = st.session_state.df_final.copy()
                        df_simpan['Series'] = final_series
                        df_simpan['Episode'] = final_episode
                        df_simpan = df_simpan.rename(columns={st.session_state.kolom_teks_aktif: "Komentar_Teks"})

                        kolom_penting = ['Series', 'Episode', 'Komentar_Teks'] + kolom_visualisasi
                        df_simpan = df_simpan[kolom_penting]

                        simpan_ke_gsheet(df_simpan)

                    st.session_state.save_status = 'success'
                    st.session_state.save_message = f"Berhasil! Data '{final_series} - {final_episode}' telah direkam abadi di Google Sheets."
                    st.rerun()

# ------------------------------------------------------------------------------
# TAB 2: DASHBOARD TREN & FILTER KOMENTAR
# ------------------------------------------------------------------------------
with tab_riwayat:
    st.markdown("### Pantau Dinamika Tren & Baca Komentar Penonton")

    if st.session_state.crud_notif:
        st.success(st.session_state.crud_notif)
        st.session_state.crud_notif = None

    with st.spinner('Mengambil data terbaru dari Cloud Database...'):
        df_riwayat = baca_gsheet()

    if not df_riwayat.empty and 'Series' in df_riwayat.columns:
        df_riwayat['Series'] = df_riwayat['Series'].astype(str).str.strip()
        df_riwayat['Episode'] = df_riwayat['Episode'].astype(str).str.strip()

        daftar_series = df_riwayat['Series'].unique().tolist()
        pilihan_series = st.selectbox("Pilih Judul Series untuk dianalisis:", daftar_series)

        if pilihan_series:
            data_terpilih = df_riwayat[df_riwayat['Series'] == pilihan_series].copy()

            col_ep, col_kat = st.columns(2)

            def urutkan_episode(ep_str):
                try:
                    return int(str(ep_str).replace("Episode ", "").strip())
                except:
                    return 999

            daftar_ep_mentah = data_terpilih['Episode'].unique().tolist()
            daftar_ep_urut = sorted(daftar_ep_mentah, key=urutkan_episode)

            with col_ep:
                filter_ep = st.selectbox("Filter Episode:", ["Semua Episode"] + daftar_ep_urut)
            with col_kat:
                filter_kat = st.selectbox("Filter Kategori Spesifik:", ["Semua Kategori"] + kolom_visualisasi)

            with st.spinner(f'Memuat data "{pilihan_series}" ({filter_ep})...'):
                df_tampil = data_terpilih.copy()
                if filter_ep != "Semua Episode":
                    df_tampil = df_tampil[df_tampil['Episode'] == filter_ep]

                if filter_kat != "Semua Kategori":
                    df_tampil = df_tampil[df_tampil[filter_kat] == 1]

                for col in kolom_visualisasi:
                    df_tampil[col] = pd.to_numeric(df_tampil[col], errors='coerce').fillna(0).astype(int)

                hitungan_kategori = df_tampil[kolom_visualisasi].sum()

                fig_riwayat, ax_riwayat = plt.subplots(figsize=(12, 6))

                if filter_ep == "Semua Episode":
                    grafik_data = df_tampil.groupby('Episode')[kolom_visualisasi].sum().reset_index()

                    grafik_data['Sort_Key'] = grafik_data['Episode'].apply(urutkan_episode)
                    grafik_data = grafik_data.sort_values('Sort_Key').drop(columns=['Sort_Key'])

                    grafik_melted = grafik_data.melt(
                        id_vars=['Episode'], value_vars=kolom_visualisasi,
                        var_name='Kategori', value_name='Total Komentar'
                    )

                    sns.barplot(data=grafik_melted, x='Episode', y='Total Komentar', hue='Kategori', palette='tab10', ax=ax_riwayat)
                    ax_riwayat.set_title(f'Distribusi Tren - {pilihan_series}', pad=15, fontweight='bold', fontsize=14)
                else:
                    sns.barplot(x=hitungan_kategori.index, y=hitungan_kategori.values, palette='tab10', ax=ax_riwayat)
                    ax_riwayat.set_title(f'Fokus Topik - {filter_ep}', pad=15, fontweight='bold', fontsize=14)

                ax_riwayat.set_xlabel('Kategori / Episode', fontweight='bold')
                ax_riwayat.set_ylabel('Jumlah Komentar', fontweight='bold')
                plt.xticks(rotation=45, ha='right')
                if filter_ep == "Semua Episode":
                    plt.legend(title='Kategori', bbox_to_anchor=(1.02, 1), loc='upper left')
                plt.tight_layout()

                if filter_kat == "Semua Kategori":
                    kolom_tampil = ['Episode', 'Komentar_Teks'] + kolom_visualisasi
                else:
                    kolom_tampil = ['Episode', 'Komentar_Teks', filter_kat]

            st.success(f"✅ Data '{pilihan_series}' ({filter_ep}) berhasil dimuat!")

            st.markdown("---")
            st.markdown(f"### 📊 Ringkasan Jumlah Komentar: {filter_ep}")

            m_cols = st.columns(len(kolom_visualisasi))
            for idx, kat in enumerate(kolom_visualisasi):
                nama_label_bersih = kat.replace("_", " ")
                m_cols[idx].metric(label=nama_label_bersih, value=int(hitungan_kategori[kat]))

            st.markdown("---")
            if filter_ep == "Semua Episode":
                st.markdown(f"**Grafik Perjalanan Tren: {pilihan_series}**")
            else:
                st.markdown(f"**Distribusi Topik: {pilihan_series} ({filter_ep})**")
            st.pyplot(fig_riwayat)

            st.markdown("---")
            st.markdown(f"**Tabel Eksplorasi Komentar: {pilihan_series}**")

            st.dataframe(df_tampil[kolom_tampil], use_container_width=True)
            st.info(f"Ditemukan {len(df_tampil)} komentar sesuai filter di atas.")

            st.markdown("---")

            with st.expander("⚙️ Manajemen & Edit Data Database", expanded=False):
                aksi_manajemen = st.radio("Pilih Tindakan:", ["✏️ Edit Data (Typo/Ubah Nama)", "🗑️ Hapus Data"], horizontal=True)

                if aksi_manajemen == "✏️ Edit Data (Typo/Ubah Nama)":
                    st.info("Fitur ini digunakan jika ada kesalahan ketik (typo) pada nama Series atau nomor Episode.")

                    label_semua_edit = f"Semua Episode {pilihan_series} (Ganti Nama Series Saja)"
                    opsi_edit = st.selectbox(
                        "Pilih data yang ingin diedit:",
                        [label_semua_edit] + daftar_ep_urut
                    )

                    col_ed1, col_ed2 = st.columns(2)
                    with col_ed1:
                        series_baru = st.text_input("Nama Series Baru:", value=pilihan_series)

                    with col_ed2:
                        if opsi_edit != label_semua_edit:
                            angka_lama = str(opsi_edit).replace("Episode ", "")
                            ep_baru = st.text_input("Nomor Episode Baru (Ketik Angkanya Saja):", value=angka_lama)
                        else:
                            st.write("")
                            st.write("")
                            ep_baru = None

                    if st.button("💾 Simpan Perubahan Data", use_container_width=True):
                        with st.spinner("Memperbarui data di Cloud..."):
                            time.sleep(0.5)
                            df_riwayat_baru = df_riwayat.copy()

                            if opsi_edit == label_semua_edit:
                                df_riwayat_baru.loc[df_riwayat_baru['Series'] == pilihan_series, 'Series'] = series_baru
                                st.session_state.crud_notif = f"Nama Series berhasil diubah dari '{pilihan_series}' menjadi '{series_baru}'!"
                            else:
                                mask = (df_riwayat_baru['Series'] == pilihan_series) & (df_riwayat_baru['Episode'] == opsi_edit)
                                df_riwayat_baru.loc[mask, 'Series'] = series_baru
                                df_riwayat_baru.loc[mask, 'Episode'] = f"Episode {ep_baru.strip()}"
                                st.session_state.crud_notif = f"Data '{pilihan_series} - {opsi_edit}' berhasil diperbarui!"

                            tulis_ulang_gsheet(df_riwayat_baru)
                            st.rerun()

                elif aksi_manajemen == "🗑️ Hapus Data":
                    st.warning("Hati-hati! Data yang dihapus dari Google Sheets tidak dapat dikembalikan.")

                    label_semua_hapus = f"Semua Episode {pilihan_series} (Hapus Seluruh Series Ini)"
                    opsi_hapus = st.selectbox(
                        "Pilih rentang data yang ingin dihapus:",
                        [label_semua_hapus] + daftar_ep_urut
                    )

                    if st.button("🗑️ Hapus Data Terpilih", use_container_width=True):
                        with st.spinner("Menghapus data di Cloud..."):
                            time.sleep(0.5)
                            if opsi_hapus == label_semua_hapus:
                                df_riwayat_baru = df_riwayat[df_riwayat['Series'] != pilihan_series]
                                tulis_ulang_gsheet(df_riwayat_baru)
                                st.session_state.crud_notif = f"Seluruh data '{pilihan_series}' berhasil dibantai dari Google Sheets!"
                            else:
                                kondisi_hapus = (df_riwayat['Series'] == pilihan_series) & (df_riwayat['Episode'] == opsi_hapus)
                                df_riwayat_baru = df_riwayat[~kondisi_hapus]
                                tulis_ulang_gsheet(df_riwayat_baru)
                                st.session_state.crud_notif = f"Data '{pilihan_series} - {opsi_hapus}' berhasil dihapus dari Google Sheets!"

                            st.rerun()
    else:
        st.info("Belum ada data yang tersimpan di Cloud Database. Silakan analisis video di tab sebelah dan klik 'Simpan'.")