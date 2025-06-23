import streamlit as st
import numpy as np
import cv2
import movement_detector
from PIL import Image
import tempfile
import os
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

st.markdown(
    """
    <style>
    body {
        background-color: #ffe4ec !important;
        color: #8B008B !important;
    }
    .stApp {
        background-color: #ffe4ec;
        color: #8B008B;
    }
    h1, h2, h3, p, label, span {
        color: #8B008B !important;
    }
    /* Analiz tab'ı için özel stil */
    .stRadio > label {
        color: #8B008B !important;
    }
    .stSlider > label {
        color: #8B008B !important;
    }
    .stCheckbox > label {
        color: #8B008B !important;
    }
    .stButton > button {
        color: #8B008B !important;
    }
    .stFileUploader > label {
        color: #8B008B !important;
    }
    /* Metric widget'ları için mor renk */
    .stMetric > div > div > div {
        color: #8B008B !important;
    }
    .stMetric > div > div > div > div {
        color: #8B008B !important;
    }
    .stMetric > div > div > div > div > div {
        color: #8B008B !important;
    }
    /* Daha spesifik metric stilleri */
    .stMetric [data-testid="metric-container"] {
        color: #8B008B !important;
    }
    .stMetric [data-testid="metric-container"] * {
        color: #8B008B !important;
    }
    .stMetric label {
        color: #8B008B !important;
    }
    .stMetric div {
        color: #8B008B !important;
    }
    .stMetric span {
        color: #8B008B !important;
    }
    .stMetric p {
        color: #8B008B !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "<h1 style='color:#8B008B; text-align:center;'>🎥 Movement Detection Demo 🎉</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div style='text-align:center; font-size:20px; color:#8B008B;'>"
    "Detect camera movement with AI!<br>"
    "Choose SIFT or ORB algorithm, play with parameters and see the results! 🚀"
    "</div>",
    unsafe_allow_html=True,
)

tab1, tab2, tab3 = st.tabs(["ℹ️ About", "📹 Upload & Analysis", "📊 Results & Visualizations"])

with tab1:
    st.subheader("👩‍💻 About Developer")
    st.info(
        "This application was developed by me: **Şeyma**. "
        "Hi, I mean just hire me! "
        "For more information: [GitHub](https://github.com/SeymaErtugrul) or [LinkedIn](https://www.linkedin.com/in/seyma-ertugrul-18b1aa199/)"
    )

    st.subheader("ℹ️ About Application")
    st.success(
        "This application detects camera movement in the video file you upload. "
        "You can choose SIFT or ORB algorithm and adjust threshold and other parameters. "
        "Moving frames and scores are displayed on screen!"
    )
    
    st.subheader("🔧 How It Works")
    st.markdown("""
    * UPLOAD VIDEO OR IMAGES
    """)

with tab2:
    upload_option = st.radio("Choose Upload Type", ["📹 Video Upload", "🖼️ Image Upload"], horizontal=True)

    if upload_option == "📹 Video Upload":
        st.subheader("📹 Upload Video")
        video_file = st.file_uploader("Select a video file", type=["mp4", "avi", "mov"])
        image_file1 = None
        image_file2 = None
    else:
        st.subheader("🖼️ Upload Images")
        col1, col2 = st.columns(2)
        with col1:
            image_file1 = st.file_uploader("Select first image", type=["jpg", "jpeg", "png"])
        with col2:
            image_file2 = st.file_uploader("Select second image", type=["jpg", "jpeg", "png"])
        video_file = None

    st.subheader("⚙️ Algorithm and Parameters")
    method = st.radio("Algorithm Selection", ["SIFT", "ORB"], horizontal=True)
    threshold = st.slider("Movement Threshold", 0.1, 2.0, 0.5, 0.05)
    min_match_count = st.slider(
        "Minimum Match Count", 2, 20, 10 if method == "SIFT" else 4
    )
    max_frames = st.slider("Maximum Frames to Analyze", 10, 300, 50, 10)
    debug = st.checkbox("Debug Mode", value=False)

    start_analysis = st.button("🚦 Start Analysis")
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None

    if start_analysis:
        if upload_option == "📹 Video Upload":
            if video_file is None:
                st.warning("Please upload a video file!")
            else:
                st.info("Processing video, please wait... ⏳")
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tfile.write(video_file.read())
                tfile.close()
                cap = cv2.VideoCapture(tfile.name)
                ret, prev_frame = cap.read()
                if not ret:
                    st.error("Could not read first frame from video.")
                else:
                    detector = movement_detector.CameraMovementDetector(
                        method=method, threshold=threshold, min_match_count=min_match_count, debug=debug
                    )
                    frame_idx = 1
                    movement_frames = []
                    movement_scores = []
                    details_list = []
                    frames = [prev_frame]
                    while True:
                        if frame_idx > max_frames:
                            break
                        ret, frame = cap.read()
                        if not ret:
                            break
                        is_movement, score, details = detector.detect(prev_frame, frame)
                        if is_movement:
                            movement_frames.append(frame_idx)
                            movement_scores.append(score)
                            details_list.append(details)
                        frames.append(frame)
                        prev_frame = frame
                        frame_idx += 1
                    cap.release()
                    os.unlink(tfile.name)
                    st.success(f"Analysis completed! 🚀 Movement detected in {len(movement_frames)} frames.")
                    st.session_state.analysis_results = {
                        'type': 'video',
                        'movement_frames': movement_frames,
                        'movement_scores': movement_scores,
                        'details_list': details_list,
                        'frames': frames,
                        'total_frames': frame_idx - 1,
                        'method': method,
                        'threshold': threshold
                    }
                    
                    if len(movement_frames) > 0:
                        st.balloons()
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.markdown(
                            f'<div style="text-align: center; padding: 10px; border: 2px solid #8B008B; border-radius: 10px; background-color: rgba(139, 0, 139, 0.1);">'
                            f'<h3 style="color: #8B008B; margin: 0;">Total Frames</h3>'
                            f'<p style="color: #8B008B; font-size: 24px; font-weight: bold; margin: 5px 0;">{frame_idx-1}</p>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                    with col2:
                        st.markdown(
                            f'<div style="text-align: center; padding: 10px; border: 2px solid #8B008B; border-radius: 10px; background-color: rgba(139, 0, 139, 0.1);">'
                            f'<h3 style="color: #8B008B; margin: 0;">Movement Frames</h3>'
                            f'<p style="color: #8B008B; font-size: 24px; font-weight: bold; margin: 5px 0;">{len(movement_frames)}</p>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                    with col3:
                        st.markdown(
                            f'<div style="text-align: center; padding: 10px; border: 2px solid #8B008B; border-radius: 10px; background-color: rgba(139, 0, 139, 0.1);">'
                            f'<h3 style="color: #8B008B; margin: 0;">Movement Rate</h3>'
                            f'<p style="color: #8B008B; font-size: 24px; font-weight: bold; margin: 5px 0;">{(len(movement_frames)/(frame_idx-1)*100):.1f}%</p>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                    with col4:
                        avg_score = np.mean(movement_scores) if movement_scores else 0.0
                        st.markdown(
                            f'<div style="text-align: center; padding: 10px; border: 2px solid #8B008B; border-radius: 10px; background-color: rgba(139, 0, 139, 0.1);">'
                            f'<h3 style="color: #8B008B; margin: 0;">Avg Score</h3>'
                            f'<p style="color: #8B008B; font-size: 24px; font-weight: bold; margin: 5px 0;">{avg_score:.3f}</p>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                    
                    st.info("📊 Check the 'Results & Visualizations' tab for detailed graphs and analysis!")
        
        else:  
            if image_file1 is None or image_file2 is None:
                st.warning("Please upload both images!")
            else:
                st.info("Processing images, please wait... ⏳")
                image1 = Image.open(image_file1)
                image2 = Image.open(image_file2)
                frame1 = np.array(image1)
                frame2 = np.array(image2)
                
                if frame1.shape[-1] == 4:
                    frame1 = frame1[:, :, :3]
                if frame2.shape[-1] == 4:
                    frame2 = frame2[:, :, :3]
        
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image1, caption="First Image", use_column_width=True)
                with col2:
                    st.image(image2, caption="Second Image", use_column_width=True)
            
                detector = movement_detector.CameraMovementDetector(
                    method=method, threshold=threshold, min_match_count=min_match_count, debug=debug
                )
                is_movement, score, details = detector.detect(frame1, frame2)
                
                st.success("Image analysis completed! 🚀")
                if is_movement:
                    st.balloons()

                st.session_state.analysis_results = {
                    'type': 'image',
                    'is_movement': is_movement,
                    'score': score,
                    'details': details,
                    'threshold': threshold,
                    'method': method
                }
                
                st.info("📊 Check the 'Results & Visualizations' tab for detailed graphs and analysis!")

with tab3:
    st.subheader("📊 Analysis Results & Visualizations")
    
    if st.session_state.analysis_results is None:
        st.info("Please run an analysis first in the 'Upload & Analysis' tab!")
    else:
        results = st.session_state.analysis_results
        
        if results['type'] == 'video':
            movement_frames = results['movement_frames']
            movement_scores = results['movement_scores']
            details_list = results['details_list']
            frames = results['frames']
            total_frames = results['total_frames']
            
            if len(movement_frames) > 0:
                all_scores = []
                all_frames = []
                for i in range(1, total_frames + 1):
                    if i in movement_frames:
                        idx = movement_frames.index(i)
                        all_scores.append(movement_scores[idx])
                    else:
                        all_scores.append(0.0)
                    all_frames.append(i)
                
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(
                    x=all_frames,
                    y=all_scores,
                    mode='lines+markers',
                    name='Movement Score',
                    line=dict(color='#8B008B', width=2),
                    marker=dict(size=6)
                ))
                fig1.update_layout(
                    title='Movement Scores Over Time',
                    xaxis_title='Frame Number',
                    yaxis_title='Movement Score',
                    template='plotly_white',
                    height=400
                )
                st.plotly_chart(fig1, use_container_width=True)
                if len(movement_scores) > 1:
                    fig2 = px.histogram(
                        x=movement_scores,
                        nbins=10,
                        title='Distribution of Movement Scores',
                        labels={'x': 'Movement Score', 'y': 'Frequency'},
                        color_discrete_sequence=['#8B008B']
                    )
                    fig2.update_layout(height=400)
                    st.plotly_chart(fig2, use_container_width=True)
                
                if len(details_list) > 0:
                    translations = [details.get('translation', 0) for details in details_list]
                    determinants = [details.get('determinant', 0) for details in details_list]
                    
                    fig3 = go.Figure()
                    fig3.add_trace(go.Scatter(
                        x=translations,
                        y=determinants,
                        mode='markers',
                        marker=dict(
                            size=10,
                            color=movement_scores,
                            colorscale='Viridis',
                            showscale=True,
                            colorbar=dict(title="Movement Score")
                        ),
                        text=[f"Frame {frame}" for frame in movement_frames],
                        hovertemplate='Frame: %{text}<br>Translation: %{x:.3f}<br>Determinant: %{y:.3f}<extra></extra>'
                    ))
                    fig3.update_layout(
                        title='Translation vs Determinant Analysis',
                        xaxis_title='Translation',
                        yaxis_title='Determinant',
                        template='plotly_white',
                        height=400
                    )
                    st.plotly_chart(fig3, use_container_width=True)

                st.subheader("🎬 Frames with Detected Movement")
                for idx, frame_idx in enumerate(movement_frames):
                    st.markdown(
                        f"<div style='border:2px solid #8B008B; border-radius:10px; padding:10px; margin-bottom:10px;'>"
                        f"<b>Movement Detected at Frame:</b> {frame_idx} <br>"
                        f"<b>Score:</b> {movement_scores[idx]:.3f} <br>"
                        f"<b>Details:</b> {details_list[idx]}"
                        "</div>",
                        unsafe_allow_html=True,
                    )
                    st.image(cv2.cvtColor(frames[frame_idx], cv2.COLOR_BGR2RGB), use_column_width=False, width=300)
            else:
                st.info("No movement detected.")
        
        else:
            is_movement = results['is_movement']
            score = results['score']
            details = results['details']
            threshold = results['threshold']
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = score,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Movement Score"},
                delta = {'reference': threshold},
                gauge = {
                    'axis': {'range': [None, max(score * 1.5, threshold * 2)]},
                    'bar': {'color': "#8B008B"},
                    'steps': [
                        {'range': [0, threshold], 'color': "lightgray"},
                        {'range': [threshold, max(score * 1.5, threshold * 2)], 'color': "lightcoral"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': threshold
                    }
                }
            ))
            fig_gauge.update_layout(height=400)
            st.plotly_chart(fig_gauge, use_container_width=True)
    
            if 'translation' in details and 'determinant' in details and 'identity_diff' in details:
                detail_names = ['Translation', 'Determinant', 'Identity Diff']
                detail_values = [details['translation'], abs(details['determinant']), details['identity_diff']]
                
                fig_bar = go.Figure(data=[
                    go.Bar(x=detail_names, y=detail_values, marker_color='#8B008B')
                ])
                fig_bar.update_layout(
                    title='Movement Components Analysis',
                    xaxis_title='Component',
                    yaxis_title='Value',
                    height=400
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            st.markdown(
                f"<div style='border:2px solid #8B008B; border-radius:10px; padding:10px; margin-bottom:10px;'>"
                f"<b>Movement Detected:</b> {is_movement} <br>"
                f"<b>Score:</b> {score:.3f} <br>"
                f"<b>Details:</b> {details}"
                "</div>",
                unsafe_allow_html=True,
            )
