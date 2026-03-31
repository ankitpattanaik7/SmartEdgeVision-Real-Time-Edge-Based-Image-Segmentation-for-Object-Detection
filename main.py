"""Streamlit app: Advanced Feature Extraction and Image Segmentation using Canny."""

from __future__ import annotations

from time import perf_counter

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from canny import custom_canny, opencv_canny
from segmentation import contour_segmentation, region_growing_segmentation, watershed_segmentation
from utils import bgr_to_rgb, edge_density, encode_png, normalize_to_uint8, performance_dataframe, read_uploaded_image, stage_figure

try:
    import av
    from streamlit_webrtc import VideoProcessorBase, webrtc_streamer

    WEBRTC_AVAILABLE = True
except Exception:
    WEBRTC_AVAILABLE = False


st.set_page_config(
    page_title="Advanced Canny + Segmentation Lab",
    page_icon="CV",
    layout="wide",
)

st.title("Advanced Feature Extraction and Image Segmentation")
st.caption("Postgraduate-level interactive study of Canny edge detection and segmentation behavior")


if WEBRTC_AVAILABLE:

    class EdgeVideoProcessor(VideoProcessorBase):
        """Apply selected edge detector frame-by-frame for webcam streaming."""

        def __init__(self, mode: str, low: int, high: int):
            self.mode = mode
            self.low = low
            self.high = high

        def recv(self, frame):
            image_bgr = frame.to_ndarray(format="bgr24")
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

            if self.mode == "Custom Canny":
                edges = custom_canny(
                    gray,
                    low_threshold=self.low,
                    high_threshold=self.high,
                    kernel_size=5,
                    sigma=1.2,
                    return_intermediates=False,
                ).edges
            else:
                edges = opencv_canny(gray, low_threshold=self.low, high_threshold=self.high)

            overlay = image_bgr.copy()
            overlay[edges > 0] = [0, 255, 0]
            return av.VideoFrame.from_ndarray(overlay, format="bgr24")


def run_detector(gray: np.ndarray, method: str, low: int, high: int, ksize: int, sigma: float):
    """Execute chosen edge detector and return edge map, intermediates, and runtime."""
    start = perf_counter()
    if method == "Custom Canny":
        result = custom_canny(
            gray,
            low_threshold=low,
            high_threshold=high,
            kernel_size=ksize,
            sigma=sigma,
            return_intermediates=True,
        )
        edges = result.edges
    else:
        result = None
        edges = opencv_canny(gray, low_threshold=low, high_threshold=high)
    elapsed_ms = (perf_counter() - start) * 1000
    return edges, result, elapsed_ms


def benchmark_detectors(gray: np.ndarray, low: int, high: int, ksize: int, sigma: float):
    """Time both implementations for side-by-side performance comparison."""
    rows = []

    t0 = perf_counter()
    custom_canny(
        gray,
        low_threshold=low,
        high_threshold=high,
        kernel_size=ksize,
        sigma=sigma,
        return_intermediates=False,
    )
    custom_ms = (perf_counter() - t0) * 1000

    t1 = perf_counter()
    opencv_canny(gray, low_threshold=low, high_threshold=high)
    opencv_ms = (perf_counter() - t1) * 1000

    rows.append(
        {
            "Method": "Custom Canny",
            "Runtime (ms)": round(custom_ms, 3),
            "Asymptotic Complexity": "O(HW) per stage; overall O(HW)",
        }
    )
    rows.append(
        {
            "Method": "OpenCV Canny",
            "Runtime (ms)": round(opencv_ms, 3),
            "Asymptotic Complexity": "O(HW) (optimized C/C++ implementation)",
        }
    )
    return performance_dataframe(rows)


def sensitivity_analysis(gray: np.ndarray, method: str, low: int, high: int, ksize: int, sigma: float):
    """Probe threshold sensitivity around selected operating point."""
    deltas = [-40, -20, 0, 20, 40]
    labels = []
    densities = []
    runtimes = []

    for d in deltas:
        l = int(np.clip(low + d, 0, 255))
        h = int(np.clip(high + d, 0, 255))
        if l > h:
            l, h = h, l

        t0 = perf_counter()
        if method == "Custom Canny":
            edge_map = custom_canny(
                gray,
                low_threshold=l,
                high_threshold=h,
                kernel_size=ksize,
                sigma=sigma,
                return_intermediates=False,
            ).edges
        else:
            edge_map = opencv_canny(gray, low_threshold=l, high_threshold=h)
        rt = (perf_counter() - t0) * 1000

        labels.append(f"({l},{h})")
        densities.append(edge_density(edge_map))
        runtimes.append(rt)

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(labels, densities, marker="o", color="tab:blue", label="Edge Density")
    ax1.set_ylabel("Edge Density", color="tab:blue")
    ax1.set_xlabel("(Low, High) Threshold Pair")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(labels, runtimes, marker="s", color="tab:red", label="Runtime (ms)")
    ax2.set_ylabel("Runtime (ms)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    fig.tight_layout()
    return fig


def render_stage_explanations():
    """Theory summary for each Canny stage."""
    with st.expander("Theoretical explanation of Canny stages", expanded=False):
        st.markdown(
            """
1. Gaussian smoothing: suppresses high-frequency noise to improve gradient stability.
2. Sobel gradients: estimates local intensity derivatives and edge orientation.
3. Non-maximum suppression: enforces one-pixel-thin ridge responses.
4. Double thresholding: separates reliable edges (strong) from candidates (weak).
5. Hysteresis tracking: preserves weak edges only if connected to strong edges.
            """
        )


with st.sidebar:
    st.header("Controls")
    method = st.radio("Edge extractor", ["Custom Canny", "OpenCV Canny"], index=0)
    low_threshold = st.slider("Low threshold", 0, 255, 50)
    high_threshold = st.slider("High threshold", 0, 255, 150)
    kernel_size = st.slider("Gaussian kernel size", 3, 11, 5, step=2)
    sigma = st.slider("Gaussian sigma", 0.5, 3.0, 1.4, step=0.1)
    min_contour_area = st.slider("Minimum contour area", 10, 2000, 120)
    region_tol = st.slider("Region-growing tolerance", 1, 60, 15)
    region_seed_mode = st.selectbox("Region seed", ["Image center", "Strongest edge location"])
    show_sensitivity = st.checkbox("Show parameter sensitivity analysis", value=True)


pipeline_tab, webcam_tab = st.tabs(["Image Pipeline", "Webcam Edge Detection"])


with pipeline_tab:
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"])

    if uploaded_file is None:
        st.info("Upload an image to start the edge extraction and segmentation pipeline.")
    else:
        image_bgr = read_uploaded_image(uploaded_file.read())
        image_rgb = bgr_to_rgb(image_bgr)
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        edge_map, canny_result, runtime_ms = run_detector(
            gray,
            method=method,
            low=low_threshold,
            high=high_threshold,
            ksize=kernel_size,
            sigma=sigma,
        )

        contour_out = contour_segmentation(image_bgr, edge_map, min_area=min_contour_area)

        if region_seed_mode == "Strongest edge location":
            ys, xs = np.where(edge_map > 0)
            if len(xs) > 0:
                seed = (int(np.median(ys)), int(np.median(xs)))
            else:
                seed = (gray.shape[0] // 2, gray.shape[1] // 2)
        else:
            seed = (gray.shape[0] // 2, gray.shape[1] // 2)

        region_out = region_growing_segmentation(image_bgr, seed_point=seed, tolerance=region_tol)
        water_out = watershed_segmentation(image_bgr)

        st.subheader("Primary outputs")
        c1, c2, c3 = st.columns(3)
        c1.image(image_rgb, caption="Original image", use_container_width=True)
        c2.image(edge_map, caption=f"Edge map ({method})", use_container_width=True)
        c3.image(
            bgr_to_rgb(contour_out["annotated"]),
            caption="Contour-based segmentation",
            use_container_width=True,
        )

        st.metric("Selected edge detector runtime", f"{runtime_ms:.3f} ms")

        st.subheader("Segmentation comparison")
        s1, s2, s3 = st.columns(3)
        s1.image(contour_out["mask"], caption="Contour mask", use_container_width=True)
        s2.image(region_out["mask"], caption="Region-growing mask", use_container_width=True)
        s3.image(water_out["markers"], caption="Watershed markers", use_container_width=True)

        p1, p2 = st.columns(2)
        p1.image(bgr_to_rgb(region_out["segmented"]), caption="Region-growing segmented", use_container_width=True)
        p2.image(bgr_to_rgb(water_out["boundaries"]), caption="Watershed boundaries", use_container_width=True)

        st.subheader("Intermediate Canny stages")
        if canny_result is not None:
            stages = [
                normalize_to_uint8(canny_result.blurred),
                normalize_to_uint8(canny_result.gradient_magnitude),
                normalize_to_uint8(canny_result.non_max_suppressed),
            ]
            titles = ["Gaussian blurred", "Gradient magnitude", "Non-max suppression"]
            fig = stage_figure(stages, titles)
            st.pyplot(fig)
            render_stage_explanations()
        else:
            st.info("Intermediate internal stages are exposed for the custom Canny mode.")

        st.subheader("Performance comparison")
        bench_df = benchmark_detectors(
            gray,
            low=low_threshold,
            high=high_threshold,
            ksize=kernel_size,
            sigma=sigma,
        )
        st.dataframe(bench_df, use_container_width=True)

        st.subheader("Download processed outputs")
        d1, d2, d3 = st.columns(3)
        d1.download_button(
            "Download edge map (PNG)",
            data=encode_png(edge_map),
            file_name="edge_map.png",
            mime="image/png",
        )
        d2.download_button(
            "Download contour segmentation (PNG)",
            data=encode_png(contour_out["annotated"], is_bgr=True),
            file_name="contour_segmentation.png",
            mime="image/png",
        )
        d3.download_button(
            "Download watershed boundaries (PNG)",
            data=encode_png(water_out["boundaries"], is_bgr=True),
            file_name="watershed_boundaries.png",
            mime="image/png",
        )

        if show_sensitivity:
            st.subheader("Parameter sensitivity analysis")
            fig_sens = sensitivity_analysis(
                gray,
                method=method,
                low=low_threshold,
                high=high_threshold,
                ksize=kernel_size,
                sigma=sigma,
            )
            st.pyplot(fig_sens)


with webcam_tab:
    st.write("Live webcam overlay with edge responses.")
    if WEBRTC_AVAILABLE:
        webrtc_streamer(
            key="edge-webcam",
            video_processor_factory=lambda: EdgeVideoProcessor(method, low_threshold, high_threshold),
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        st.caption("Green pixels indicate detected edges in the live stream.")
    else:
        st.warning(
            "Webcam real-time mode requires streamlit-webrtc and av. Install dependencies from requirements.txt."
        )
