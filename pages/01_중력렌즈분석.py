import streamlit as st
import numpy as np
import cv2
from matplotlib import pyplot as plt
from astropy.io import fits
from PIL import Image
import io

st.set_page_config(page_title="중력렌즈 왜곡 분석기", layout="wide")

st.title("🔭 중력렌즈 왜곡 구조 분석 앱")
st.markdown("""
천체 이미지에서 중력렌즈로 인한 왜곡 구조(호, 다중상 등)를 시각화하고 분석합니다.
- 밝기 정규화
- 왜곡 강조 필터(Canny, Laplacian)
- 비대칭성 중심 시각화
""")

# 업로드
uploaded_file = st.file_uploader("이미지를 업로드하세요 (FITS, PNG, JPG)", type=["fits", "png", "jpg", "jpeg"])

if uploaded_file is not None:
    filename = uploaded_file.name

    if filename.endswith(".fits"):
        with fits.open(uploaded_file) as hdul:
            data = hdul[0].data
            image = np.nan_to_num(data)
            image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255
            image = image.astype(np.uint8)
    else:
        pil_img = Image.open(uploaded_file).convert("L")
        image = np.array(pil_img)

    st.subheader("📷 원본 이미지")
    st.image(image, use_column_width=True, clamp=True)

    # 밝기 히스토그램
    st.subheader("📊 밝기 히스토그램")
    fig1, ax1 = plt.subplots()
    ax1.hist(image.ravel(), bins=256, color='orange', alpha=0.7)
    ax1.set_xlabel("픽셀 밝기")
    ax1.set_ylabel("빈도")
    st.pyplot(fig1)

    # 엣지 분석
    st.subheader("🔍 왜곡 구조 강조")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Canny 엣지 검출")
        edges = cv2.Canny(image, 30, 120)
        st.image(edges, use_column_width=True, clamp=True)

    with col2:
        st.write("Laplacian 필터")
        lap = cv2.Laplacian(image, cv2.CV_64F)
        lap = np.uint8(np.absolute(lap))
        st.image(lap, use_column_width=True, clamp=True)

    # 중심 대칭 분석
    st.subheader("📌 밝기 중심과 비대칭 구조 시각화")

    # 밝기 중심 좌표
    Y, X = np.indices(image.shape)
    total = image.sum()
    x_center = int((X * image).sum() / total)
    y_center = int((Y * image).sum() / total)

    # 시각화용 마커 추가
    marked = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.circle(marked, (x_center, y_center), 5, (0, 0, 255), -1)

    st.image(marked, caption="붉은 점: 밝기 중심 (중력렌즈의 중심일 가능성)", use_column_width=True)

    st.info("엣지 구조가 중심을 기준으로 비대칭적이면 중력렌즈 의심 구조일 가능성이 있습니다.")
