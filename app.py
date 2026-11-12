"""
Dashboard Statistik Deskriptif - Student Depression Dataset
Statistik dan Probabilitas 1

Struktur folder:
    app.py
    requirements.txt
    data/
        Student_Depression_Dataset.csv

Jalankan dari terminal (di dalam folder ini):
    python -m streamlit run app.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# ======================================================================
# 1. KONFIGURASI HALAMAN
# ======================================================================
st.set_page_config(page_title="Dashboard Student Depression", page_icon="🎓", layout="wide")

DATA_DIR = Path(__file__).parent / "data"
DATA_DEFAULT = DATA_DIR / "Student_Depression_Dataset.csv"
MAKS_KATEGORI = 15  # kolom teks dengan nilai unik <= ini dianggap kategori


# ======================================================================
# 2. FUNGSI BANTU
# ======================================================================
@st.cache_data
def muat_data_default(path):
    """Baca CSV bawaan. Di-cache supaya tidak dibaca ulang setiap interaksi."""
    return pd.read_csv(path)


def hitung_statistik(x: pd.Series) -> dict:
    """Hitung seluruh ukuran statistik deskriptif untuk satu kolom numerik."""
    q1, q3 = x.quantile(0.25), x.quantile(0.75)
    iqr = q3 - q1
    batas_bawah = q1 - 1.5 * iqr
    batas_atas = q3 + 1.5 * iqr
    return {
        "n": len(x),
        "mean": x.mean(),
        "median": x.median(),
        "modus": list(x.mode()),
        "std": x.std(),                       # sampel (ddof=1)
        "skew": x.skew(),
        "min": x.min(),
        "max": x.max(),
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "batas_bawah": batas_bawah,
        "batas_atas": batas_atas,
        "outlier": x[(x < batas_bawah) | (x > batas_atas)],
    }


def format_modus(daftar, maks_tampil=3):
    teks = ", ".join(f"{m:g}" for m in daftar[:maks_tampil])
    return teks + (" ..." if len(daftar) > maks_tampil else "")


def kolom_diskret(x: pd.Series) -> bool:
    """True kalau kolom hanya berisi sedikit nilai bulat (mis. Age, Current_semester)."""
    x = x.dropna()
    return len(x) > 0 and x.nunique() <= 20 and bool(np.allclose(x, x.round()))


# ======================================================================
# 3. SIDEBAR: SUMBER DATA, PILIH KOLOM, FILTER
# ======================================================================
st.sidebar.title("⚙️ Pengaturan")

file_upload = st.sidebar.file_uploader("Upload CSV sendiri (opsional)", type="csv")
if file_upload is not None:
    df = pd.read_csv(file_upload)
    st.sidebar.success(f"Memakai file upload: {file_upload.name}")
elif DATA_DEFAULT.exists():
    df = muat_data_default(DATA_DEFAULT)
    st.sidebar.info(f"Memakai {DATA_DEFAULT.name}")
else:
    st.title("🎓 Dashboard Student Depression")
    st.warning(f"File `{DATA_DEFAULT.name}` tidak ditemukan di folder `{DATA_DIR}`. "
               "Taruh CSV-nya di sana atau upload lewat sidebar.")
    st.stop()

kolom_numerik = df.select_dtypes(include="number").columns.tolist()
kolom_kategori = [
    k for k in df.select_dtypes(exclude="number").columns if df[k].nunique() <= MAKS_KATEGORI
]

if not kolom_numerik:
    st.error("File ini tidak punya kolom numerik. Coba upload file lain.")
    st.stop()

kolom = st.sidebar.selectbox("Kolom numerik yang dianalisis", kolom_numerik)

df_f = df.copy()
with st.sidebar.expander("Filter data", expanded=False):
    for k in kolom_kategori:
        semua = sorted(df[k].dropna().astype(str).unique())
        pilihan = st.multiselect(k, semua, default=semua, key=f"filter_{k}")
        df_f = df_f[df_f[k].astype(str).isin(pilihan)]

x = df_f[kolom].dropna()
if x.empty:
    st.warning("Tidak ada data setelah difilter. Longgarkan filter di sidebar.")
    st.stop()

stat = hitung_statistik(x)


# ======================================================================
# 4. HALAMAN UTAMA
# ======================================================================
st.title("🎓 Dashboard Statistik Deskriptif: Student Depression")
st.caption(f"Kolom: **{kolom}** | Data terpilih: {len(df_f):,} dari {len(df):,} baris")

tab_data, tab_stat, tab_vis, tab_kat, tab_sim = st.tabs(
    ["📋 Data", "🧮 Statistik", "📈 Visualisasi", "🏷️ Kategori", "🎲 Simulasi Sampling"]
)

# ----------------------------- TAB DATA -------------------------------
with tab_data:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jumlah baris", f"{len(df_f):,}")
    c2.metric("Jumlah kolom", df_f.shape[1])
    c3.metric("Data kosong (total)", f"{int(df_f.isna().sum().sum()):,}")
    if "Depression" in df_f.columns and len(df_f) > 0:
        persen = (df_f["Depression"].astype(str) == "Yes").mean() * 100
        c4.metric("Depression = Yes", f"{persen:.1f}%")

    st.dataframe(df_f, width="stretch")

    with st.expander("Lihat data kosong per kolom"):
        st.dataframe(df_f.isna().sum().rename("Jumlah kosong"))

    with st.expander("Tipe data tiap kolom"):
        st.dataframe(df_f.dtypes.astype(str).rename("Tipe"))

# --------------------------- TAB STATISTIK ----------------------------
with tab_stat:
    st.subheader("Ukuran Pemusatan")
    a, b, c, d = st.columns(4)
    a.metric("Mean", f"{stat['mean']:.2f}")
    b.metric("Median", f"{stat['median']:.2f}")
    c.metric("Modus", format_modus(stat["modus"]))
    d.metric("Jumlah data (n)", f"{stat['n']:,}")

    st.subheader("Ukuran Penyebaran")
    e, f, g, h = st.columns(4)
    e.metric("Std Dev (sampel)", f"{stat['std']:.2f}" if stat["n"] > 1 else "-")
    f.metric("Range", f"{stat['max'] - stat['min']:.2f}")
    g.metric("IQR", f"{stat['iqr']:.2f}")
    h.metric("Jumlah outlier", f"{len(stat['outlier']):,}")

    st.metric("Skewness (kemencengan)", f"{stat['skew']:.2f}" if stat["n"] > 2 else "-")

    # Interpretasi otomatis: bandingkan mean dan median
    selisih = stat["mean"] - stat["median"]
    if stat["n"] < 2 or np.isnan(stat["std"]) or stat["std"] == 0:
        st.info("Data terlalu sedikit atau tidak bervariasi untuk diinterpretasi.")
    elif abs(selisih) < 0.1 * stat["std"]:
        st.success("Mean ≈ Median → distribusi relatif **simetris**.")
    elif selisih > 0:
        st.info("Mean > Median → distribusi **condong ke kanan** (ada nilai besar yang menarik mean).")
    else:
        st.info("Mean < Median → distribusi **condong ke kiri** (ada nilai kecil yang menarik mean).")

    if kolom_diskret(x):
        st.caption("Catatan: kolom ini bernilai bulat dengan sedikit variasi, jadi modus biasanya "
                   "lebih informatif daripada mean.")

    with st.expander("Detail kuartil dan batas outlier (aturan 1,5 × IQR)"):
        st.write(f"Q1 = {stat['q1']:.2f} | Q3 = {stat['q3']:.2f}")
        st.write(f"Batas bawah = {stat['batas_bawah']:.2f} | Batas atas = {stat['batas_atas']:.2f}")
        if len(stat["outlier"]) > 0:
            st.write(f"Jumlah outlier: {len(stat['outlier']):,}. Contoh 20 nilai pertama:")
            st.write(stat["outlier"].head(20).tolist())
        else:
            st.write("Tidak ada outlier.")

    st.subheader("Statistik per kelompok")
    if kolom_kategori:
        grup_stat = st.selectbox("Kelompokkan berdasarkan", kolom_kategori, key="grup_stat")
        ringkas = (
            df_f.groupby(grup_stat)[kolom]
            .agg(n="count", mean="mean", median="median", std="std", min="min", max="max")
            .round(2)
        )
        st.dataframe(ringkas, width="stretch")
    else:
        st.info("Tidak ada kolom kategori untuk pengelompokan.")

    st.subheader("Ringkasan semua kolom numerik")
    st.dataframe(df_f[kolom_numerik].describe().round(2), width="stretch")

    if len(kolom_numerik) >= 2:
        with st.expander("Korelasi antar kolom numerik"):
            korelasi = df_f[kolom_numerik].corr().round(2)
            st.dataframe(korelasi.style.background_gradient(cmap="RdBu_r", vmin=-1, vmax=1), width="stretch")
            st.caption("Korelasi menunjukkan keterkaitan linear, bukan hubungan sebab-akibat.")

# --------------------------- TAB VISUALISASI --------------------------
with tab_vis:
    kiri, kanan = st.columns(2)

    with kiri:
        st.subheader("Histogram")
        fig, ax = plt.subplots(figsize=(6, 4))
        if kolom_diskret(x):
            tepi = np.arange(x.min() - 0.5, x.max() + 1.5, 1)
            ax.hist(x, bins=tepi, edgecolor="black")
        else:
            n_bins = st.slider("Jumlah bin", 5, 60, 20)
            ax.hist(x, bins=n_bins, edgecolor="black")
        ax.axvline(stat["mean"], color="red", linestyle="--", label=f"Mean = {stat['mean']:.1f}")
        ax.axvline(stat["median"], color="green", label=f"Median = {stat['median']:.1f}")
        ax.set_xlabel(kolom)
        ax.set_ylabel("Frekuensi")
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)

    with kanan:
        st.subheader("Boxplot")
        opsi_grup = ["(tanpa pengelompokan)"] + kolom_kategori
        idx_awal = opsi_grup.index("Depression") if "Depression" in opsi_grup else 0
        grup = st.selectbox("Kelompokkan berdasarkan", opsi_grup, index=idx_awal, key="grup_box")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        if grup == opsi_grup[0]:
            ax2.boxplot(x)
            ax2.set_xticks([1])
            ax2.set_xticklabels([kolom])
        else:
            label, data = [], []
            for nama_grup, isi in df_f.groupby(grup):
                nilai = isi[kolom].dropna()
                if len(nilai) > 0:
                    label.append(str(nama_grup))
                    data.append(nilai)
            ax2.boxplot(data)
            ax2.set_xticks(range(1, len(label) + 1))
            ax2.set_xticklabels(label, rotation=30, ha="right")
            ax2.set_xlabel(grup)
        ax2.set_ylabel(kolom)
        fig2.tight_layout()
        st.pyplot(fig2)

    st.caption("Tip: bandingkan posisi garis mean dan median pada histogram dengan bentuk boxplot.")

    if len(kolom_numerik) >= 2:
        st.divider()
        st.subheader("Scatter plot dua kolom numerik")
        s1, s2, s3 = st.columns(3)
        sumbu_x = s1.selectbox("Sumbu X", kolom_numerik, index=0, key="sc_x")
        sumbu_y = s2.selectbox("Sumbu Y", kolom_numerik, index=min(1, len(kolom_numerik) - 1), key="sc_y")
        warna_opsi = ["(tanpa warna)"] + kolom_kategori
        warna = s3.selectbox("Warnai berdasarkan", warna_opsi, key="sc_warna")

        fig5, ax5 = plt.subplots(figsize=(8, 4))
        # jitter kecil supaya titik pada kolom bulat tidak menumpuk
        rng_j = np.random.default_rng(0)

        def jit(s):
            if kolom_diskret(s):
                return s + rng_j.normal(0, 0.05 * (s.std() or 1), len(s))
            return s

        if warna == warna_opsi[0]:
            ax5.scatter(jit(df_f[sumbu_x]), jit(df_f[sumbu_y]), s=8, alpha=0.4)
        else:
            for nama_grup, isi in df_f.groupby(warna):
                ax5.scatter(jit(isi[sumbu_x]), jit(isi[sumbu_y]), s=8, alpha=0.4, label=str(nama_grup))
            ax5.legend(title=warna)
        ax5.set_xlabel(sumbu_x)
        ax5.set_ylabel(sumbu_y)
        fig5.tight_layout()
        st.pyplot(fig5)

# --------------------------- TAB KATEGORI -----------------------------
with tab_kat:
    if not kolom_kategori:
        st.info("Tidak ada kolom kategori pada data ini.")
    else:
        st.subheader("Distribusi satu kolom kategori")
        idx_awal = kolom_kategori.index("Depression") if "Depression" in kolom_kategori else 0
        kat = st.selectbox("Kolom kategori", kolom_kategori, index=idx_awal, key="kat_tunggal")
        frek = df_f[kat].value_counts()
        tabel_frek = pd.DataFrame({
            "Frekuensi": frek,
            "Persen (%)": (frek / frek.sum() * 100).round(1),
        })
        k1, k2 = st.columns([1, 1])
        with k1:
            st.dataframe(tabel_frek, width="stretch")
        with k2:
            fig6, ax6 = plt.subplots(figsize=(6, 3.5))
            ax6.bar(frek.index.astype(str), frek.values, edgecolor="black")
            ax6.set_ylabel("Frekuensi")
            ax6.tick_params(axis="x", rotation=30)
            fig6.tight_layout()
            st.pyplot(fig6)

        st.divider()
        st.subheader("Tabel silang dua kolom kategori")
        t1, t2 = st.columns(2)
        baris = t1.selectbox("Kolom baris", kolom_kategori, index=idx_awal, key="silang_baris")
        lain = [k for k in kolom_kategori if k != baris]
        if lain:
            idx_kol = lain.index("Health_condition") if "Health_condition" in lain else 0
            kolom_sil = t2.selectbox("Kolom kolom", lain, index=idx_kol, key="silang_kolom")
            silang = pd.crosstab(df_f[baris], df_f[kolom_sil])
            persen_baris = (silang.div(silang.sum(axis=1), axis=0) * 100).round(1)

            tc1, tc2 = st.columns(2)
            with tc1:
                st.caption("Frekuensi")
                st.dataframe(silang, width="stretch")
            with tc2:
                st.caption(f"Persen per baris ({baris}): tiap baris berjumlah 100%")
                st.dataframe(persen_baris, width="stretch")

            fig7, ax7 = plt.subplots(figsize=(8, 3.8))
            persen_baris.plot(kind="bar", stacked=True, ax=ax7, edgecolor="black")
            ax7.set_ylabel("Persen (%)")
            ax7.set_xlabel(baris)
            ax7.legend(title=kolom_sil, bbox_to_anchor=(1.01, 1), loc="upper left")
            ax7.tick_params(axis="x", rotation=0)
            fig7.tight_layout()
            st.pyplot(fig7)
            st.caption("Perbedaan persentase antar kelompok menggambarkan pola dalam sampel, "
                       "bukan bukti sebab-akibat.")
        else:
            st.info("Perlu minimal dua kolom kategori untuk tabel silang.")

# ------------------------ TAB SIMULASI SAMPLING -----------------------
with tab_sim:
    st.subheader("Simulasi Sampling: Teorema Limit Pusat")
    st.write(
        f"Data kolom **{kolom}** dianggap sebagai populasi. Kita ambil sampel acak berukuran *n* "
        "berulang kali, hitung rata-rata tiap sampel, lalu lihat bentuk distribusi rata-rata tersebut. "
        "Makin besar *n*, distribusinya makin mendekati normal dan makin sempit, "
        "dengan simpangan baku ≈ σ/√n."
    )

    s1, s2 = st.columns(2)
    n_sampel = s1.slider("Ukuran sampel (n)", 1, 200, 30)
    n_ulang = s2.slider("Jumlah pengulangan", 100, 5000, 1000, step=100)

    if st.button("🎲 Jalankan simulasi!"):
        rng = np.random.default_rng()
        cuplikan = rng.choice(x.to_numpy(), size=(n_ulang, n_sampel), replace=True)
        st.session_state["rata_sampel"] = cuplikan.mean(axis=1)
        st.session_state["n_terakhir"] = n_sampel
        st.session_state["kolom_simulasi"] = kolom

    rata = st.session_state.get("rata_sampel")
    if rata is None or st.session_state.get("kolom_simulasi") != kolom:
        st.info("Tekan tombol **Jalankan simulasi!** untuk memulai.")
    else:
        n_pakai = st.session_state["n_terakhir"]
        sigma = x.std(ddof=0)
        se_teori = sigma / np.sqrt(n_pakai)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Mean populasi", f"{x.mean():.2f}")
        m2.metric("Mean dari rata-rata sampel", f"{rata.mean():.2f}")
        m3.metric("Std rata-rata sampel", f"{rata.std(ddof=1):.3f}")
        m4.metric("σ/√n (teori)", f"{se_teori:.3f}")

        fig3, (ax_pop, ax_samp) = plt.subplots(1, 2, figsize=(11, 3.8), sharex=True)
        ax_pop.hist(x, bins=30, edgecolor="black")
        ax_pop.set_title("Populasi (data asli)")
        ax_pop.set_xlabel(kolom)
        ax_samp.hist(rata, bins=30, edgecolor="black", color="orange")
        ax_samp.axvline(x.mean(), color="red", linestyle="--", label="Mean populasi")
        ax_samp.set_title(f"Distribusi rata-rata sampel (n = {n_pakai})")
        ax_samp.set_xlabel(kolom)
        ax_samp.legend()
        fig3.tight_layout()
        st.pyplot(fig3)

st.divider()
st.caption("Statistik dan Probabilitas 1 - Dashboard Student Depression Dataset")