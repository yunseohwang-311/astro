import streamlit as st
import numpy as np
import cv2
from matplotlib import pyplot as plt
from astropy.io import fits
from PIL import Image
import io

st.set_page_config(page_title="중력렌즈 분석기", layout="wide")

st.title("🔭 중력렌즈 분석 앱")
st.write("이미지를 업로드하여 중력렌즈 링 구조를 확인하세요.")

uploaded_file = st.file_uploader("이미지 업로드 (FITS, JPG, PNG)", type=['fits', 'jpg', 'png'])

if uploaded_file is not None:
    file_name = uploaded_file.name

    if file_name.endswith('.fits'):
        with fits.open(uploaded_file) as hdul:
            data = hdul[0].data
            if data is not None:
                image_data = np.nan_to_num(data)
                image_data = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data))
                image_data = (image_data * 255).astype(np.uint8)
    else:
        image = Image.open(uploaded_file).convert("L")
        image_data = np.array(image)

    st.subheader("원본 이미지")
    st.image(image_data, use_column_width=True, clamp=True)

    # 분석 도구들
    st.subheader("이미지 분석")

    col1, col2 = st.columns(2)

    with col1:
        st.write("🌀 라플라시안(렌즈 구조 강조)")
        laplacian = cv2.Laplacian(image_data, cv2.CV_64F)
        laplacian = np.uint8(np.absolute(laplacian))
        st.image(laplacian, use_column_width=True, clamp=True)

    with col2:
        st.write("🧩 엣지 디텍션 (Canny)")
        edges = cv2.Canny(image_data, 50, 150)
        st.image(edges, use_column_width=True, clamp=True)

    # 히스토그램
    st.subheader("픽셀 밝기 분포 (히스토그램)")
    fig, ax = plt.subplots()
    ax.hist(image_data.ravel(), bins=256, color='skyblue', alpha=0.7)
    ax.set_xlabel("픽셀 밝기")
    ax.set_ylabel("빈도")
    st.pyplot(fig)

    st.success("간단한 분석이 완료되었습니다. 중력렌즈 의심 링이 강조되어 보이나요?")
