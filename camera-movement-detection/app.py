import plotly.graph_objects as go
import pandas as pd
import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import sys
import importlib.util

# Add current directory to path and import movement_detector
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from movement_detector import *

def load_css():
    css_path = os.path.join(os.path.dirname(__file__), "style.css")
    try:
        with open(css_path) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("⚠️ style.css file not found. Using default styles.")

def display_results_tab():
    st.subheader("📊 Analysis Results & Visualizations")
    
    if st.session_state.analysis_results is None:
        st.info("🎬 Please run an analysis first in the 'Upload & Analysis' tab!")
        return
    
    results = st.session_state.analysis_results
    
    if results['type'] == 'video':
        display_video_results(results)
    else:
        display_image_results(results)

def display_video_results(results):
    st.subheader("🎥 Video Analysis Results")
    
    if results['analysis_type'] in ["📹 Camera Only", "🔄 Both"] and results['movement_frames']:
        display_camera_results(results['movement_frames'], results['movement_scores'], results['details_list'], results['total_frames'])
    
    if results['analysis_type'] in ["🎯 Object Only", "🔄 Both"] and results['object_results']:
        display_object_results(results['object_results'])

def display_camera_results(movement_frames, movement_scores, details_list, total_frames):
    st.subheader("📹 Camera Movement Results")
    
    if len(movement_frames) > 0:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🎬 Movement Frames", len(movement_frames))
        with col2:
            st.metric("📊 Detection Rate", f"{(len(movement_frames)/total_frames)*100:.1f}%")
        with col3:
            st.metric("📈 Average Score", f"{np.mean(movement_scores):.3f}")
        with col4:
            st.metric("🏆 Max Score", f"{np.max(movement_scores):.3f}")
        
        st.subheader("📋 Movement Frame Details")
        movement_data = []
        for i, (frame, score) in enumerate(zip(movement_frames, movement_scores)):
            details = details_list[i] if i < len(details_list) else {}
            movement_data.append({
                "🎬 Frame": frame,
                "📊 Score": f"{score:.3f}",
                "🔄 Translation": f"{details.get('translation', 0):.3f}",
                "📐 Determinant": f"{details.get('determinant', 0):.3f}",
                "🆔 Identity Diff": f"{details.get('identity_diff', 0):.3f}",
                "🔗 Matches": details.get('num_matches', 0),
                "✅ Inlier Ratio": f"{details.get('inlier_ratio', 0):.3f}"
            })
        
        df = pd.DataFrame(movement_data)
        st.dataframe(df, use_container_width=True)
        
        st.subheader("📈 Movement Score Timeline")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=movement_frames,
            y=movement_scores,
            mode='lines+markers',
            name='Movement Score',
            line=dict(color='#FF6B6B', width=2),
            marker=dict(size=6, color='#FF6B6B')
        ))
        fig.update_layout(
            title="📹 Camera Movement Score Over Time",
            xaxis_title="🎬 Frame Number",
            yaxis_title="📊 Movement Score",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("😴 No camera movement detected.")

def display_object_results(object_results):
    st.subheader("🎯 Object Movement Results")
    
    object_frames = object_results.get('object_frames', [])
    object_scores = object_results.get('object_scores', [])
    total_frames = object_results.get('total_frames', 0)
    
    if len(object_frames) > 0:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🎬 Movement Frames", len(object_frames))
        with col2:
            st.metric("📊 Detection Rate", f"{(len(object_frames)/total_frames)*100:.1f}%")
        with col3:
            st.metric("📈 Average Score", f"{np.mean(object_scores):.3f}")
        with col4:
            st.metric("🏆 Max Score", f"{np.max(object_scores):.3f}")
        
        st.subheader("📋 Object Movement Frame Details")
        object_data = []
        for frame, score in zip(object_frames, object_scores):
            object_data.append({
                "🎬 Frame": frame,
                "📊 Score": f"{score:.3f}",
                "🎯 Movement Type": "Object Movement"
            })
        
        df = pd.DataFrame(object_data)
        st.dataframe(df, use_container_width=True)
        
        st.subheader("📈 Object Movement Score Timeline")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=object_frames,
            y=object_scores,
            mode='lines+markers',
            name='Object Movement Score',
            line=dict(color='#4ECDC4', width=2),
            marker=dict(size=6, color='#4ECDC4')
        ))
        fig.update_layout(
            title="🎯 Object Movement Score Over Time",
            xaxis_title="🎬 Frame Number",
            yaxis_title="📊 Movement Score",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("😴 No object movement detected.")

def display_image_results(results):
    st.subheader("🖼️ Image Analysis Results")
    
    is_movement = results.get('is_movement', False)
    score = results.get('score', 0.0)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("🎯 Movement Detected", "✅ Yes" if is_movement else "❌ No")
    with col2:
        st.metric("📊 Score", f"{score:.3f}")
    
    if is_movement:
        st.success(f"🎉 Movement detected! Score: {score:.3f}")
    else:
        st.info(f"😴 No movement detected. Score: {score:.3f}")

def setup_sidebar():
    st.sidebar.header("⚙️ Settings")
    
    analysis_type = st.sidebar.radio(
        "🎯 Analysis Type", 
        ["📹 Camera Only", "🎯 Object Only", "🔄 Both"],
        help="Select analysis type",
        key="analysis_type"
    )
    
    params = {
        'camera_method': "ORB",
        'object_method': "Lucas-Kanade",
        'threshold': 0.5,
        'min_match_count': 4,
        'camera_max_frames': 50,
        'debug': False,
        'object_max_frames': 50,
        'max_corners': 150,
        'quality_level': 0.3,
        'min_distance': 7,
        'flow_threshold': 0.5,
        'object_threshold': 0.1,
        'store_frames': True,
        'frame_skip': 1,
        'enable_live_viz': True
    }
    
    if analysis_type in ["📹 Camera Only", "🔄 Both"]:
        st.sidebar.subheader("📹 Camera")
        params['camera_method'] = st.sidebar.radio("🔧 Algorithm", ["ORB", "SIFT"], key="camera_algorithm")
        params['threshold'] = st.sidebar.slider("🎯 Threshold", 0.1, 2.0, 0.5, 0.05, key="camera_threshold")

    if analysis_type in ["🎯 Object Only", "🔄 Both"]:
        st.sidebar.subheader("🎯 Object")
        params['object_method'] = st.sidebar.radio("🔧 Algorithm", ["Lucas-Kanade", "Farneback"], key="object_algorithm")
        
        if params['object_method'] == "Lucas-Kanade":
            st.sidebar.caption("🎯 Lucas-Kanade: Detects object movement by tracking corner points")
            params['max_corners'] = st.sidebar.slider("🔢 Max Corners", 20, 200, 150, 10, key="max_corners", 
                                                    help="Number of corner points to track (lower = faster, higher = more accurate)")
            params['quality_level'] = st.sidebar.slider("⭐ Quality Level", 0.01, 0.5, 0.3, 0.01, key="quality_level",
                                                       help="Minimum quality of corner points (lower = more points, higher = better quality)")
            params['min_distance'] = st.sidebar.slider("📏 Min Distance", 3, 15, 7, 1, key="min_distance",
                                                      help="Minimum distance between corner points")
        else:
            st.sidebar.caption("🌊 Farneback: Detects object movement using dense optical flow")
            params['object_threshold'] = st.sidebar.slider("🎯 Object Threshold", 0.05, 1.0, 0.3, 0.05, key="object_threshold",
                                                          help="Threshold for object movement detection (lower = more sensitive)")
            params['flow_threshold'] = st.sidebar.slider("🌊 Flow Threshold", 0.05, 1.0, 0.3, 0.05, key="flow_threshold",
                                                        help="Threshold for optical flow magnitude (lower = more sensitive)")
        
        st.sidebar.info("""
        **🎯 Object vs Camera Movement:**
        - 🟢 Green: Object movement detected
        - 🔴 Red: Camera movement or no movement
        - 📉 Lower thresholds = more sensitive detection
        - ⚙️ Adjust parameters for your specific video
        """)

    st.sidebar.subheader("⚡ Performance")
    if analysis_type in ["📹 Camera Only", "🔄 Both"]:
        params['camera_max_frames'] = st.sidebar.slider(
            "🎬 Camera Max Frames", 10, 300, 50, 10, key="camera_max_frames",
            help="Maximum number of frames to analyze for camera movement. Lower = faster analysis."
        )
    if analysis_type in ["🎯 Object Only", "🔄 Both"]:
        params['object_max_frames'] = st.sidebar.slider(
            "🎬 Object Max Frames", 10, 300, 50, 10, key="object_max_frames",
            help="Maximum number of frames to analyze for object movement. Lower = faster analysis."
        )
    params['frame_skip'] = st.sidebar.slider("⏭️ Frame Skip", 1, 5, 1, 1, key="frame_skip")
    params['enable_live_viz'] = st.sidebar.checkbox("👁️ Live Visualization", value=True, key="live_viz")
    
    return analysis_type, params

st.set_page_config(
    page_title="Movement Detection",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

# --- Tutorial/Tour State ---
if 'show_tour' not in st.session_state:
    st.session_state.show_tour = True
if 'tour_step' not in st.session_state:
    st.session_state.tour_step = 0

tour_steps = [
    {
        'title': '👋 Welcome!',
        'content': 'With this app, you can detect movement in videos or between two images. Continue the tour to get started!'
    },
    {
        'title': '1️⃣ Select Analysis Type',
        'content': 'In the "Upload & Analysis" tab, first select the analysis type: Video or Image Comparison.'
    },
    {
        'title': '2️⃣ Upload Your File(s)',
        'content': 'Depending on your choice, upload a video or two images.'
    },
    {
        'title': '3️⃣ Configure the Settings',
        'content': 'Adjust the settings in the sidebar as needed.'
    },
    {
        'title': '4️⃣ Start the Analysis',
        'content': 'Click the Start Analysis button and view the results in the Results tab.'
    },
    {
        'title': '🎉 You Are Ready!',
        'content': 'You can now start using the application!'
    },
]

def show_tour_box():
    step = st.session_state.tour_step
    st.markdown(f"""
    <div style='background-color:#f3e6fa; padding:18px; border-radius:10px; margin-bottom:20px;'>
        <b>{tour_steps[step]['title']}</b><br>
        {tour_steps[step]['content']}
        <br><br>
        <div style='display:flex; gap:10px;'>
    """, unsafe_allow_html=True)
    col1, col2 = st.columns([1,1])
    with col1:
        if step < len(tour_steps) - 1:
            if st.button('Next', key=f'tour_next_{step}'):
                st.session_state.tour_step += 1
                st.rerun()
        else:
            if st.button('Finish', key='tour_finish'):
                st.session_state.show_tour = False
                st.rerun()
    with col2:
        if st.button('Close', key=f'tour_close_{step}'):
            st.session_state.show_tour = False
            st.rerun()
    st.markdown("</div></div>", unsafe_allow_html=True)

# --- Show Tour at the Top ---
if st.session_state.show_tour:
    show_tour_box()

load_css()

st.markdown("<h1 style='color:#8B008B; text-align:center;'>🎥 Movement Detection Demo 🎉</h1>", unsafe_allow_html=True)
st.markdown(
    "<div style='text-align:center; font-size:20px; color:#8B008B;'>"
    "Detect camera and object movement in videos and images using computer vision techniques"
    "</div>",
    unsafe_allow_html=True
)

tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload & Analysis", "📊 Results", "👩‍💻 About Developer", "🎥 About Application"])

with tab1:
    st.subheader("📤 Upload & Analysis")
    analysis_type, params = setup_sidebar()

    upload_type = st.selectbox(
        "📁 Choose Upload Type",
        ["Video File", "Image Comparison"],
        help="Select the type of file to upload for analysis"
    )

    if upload_type == "Image Comparison":
        st.subheader("🖼️ Image Comparison")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📸 First Image")
            uploaded_file1 = st.file_uploader(
                "📁 Upload first image",
                type=['jpg', 'jpeg', 'png'],
                key="image1",
                help="Upload the first image for comparison"
            )
            if uploaded_file1 is not None:
                st.image(uploaded_file1, caption="📸 First Image", use_container_width=True)
        with col2:
            st.subheader("📸 Second Image")
            uploaded_file2 = st.file_uploader(
                "📁 Upload second image",
                type=['jpg', 'jpeg', 'png'],
                key="image2",
                help="Upload the second image for comparison"
            )
            if uploaded_file2 is not None:
                st.image(uploaded_file2, caption="📸 Second Image", use_container_width=True)
        if uploaded_file1 is not None and uploaded_file2 is not None:
            col1, col2 = st.columns([1, 2])
            with col1:
                if st.button("🚀 Start Image Analysis", type="primary", use_container_width=True):
                    with st.spinner("🔍 Analyzing images..."):
                        detector = CameraMovementDetector(
                            method=params['camera_method'],
                            threshold=params['threshold'],
                            min_match_count=params['min_match_count']
                        )
                        # Save both images to temporary files
                        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{uploaded_file1.name.split(".")[-1]}') as tmp_file1:
                            tmp_file1.write(uploaded_file1.getvalue())
                            file_path1 = tmp_file1.name
                        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{uploaded_file2.name.split(".")[-1]}') as tmp_file2:
                            tmp_file2.write(uploaded_file2.getvalue())
                            file_path2 = tmp_file2.name
                        # Load images and detect movement
                        image1 = cv2.imread(file_path1)
                        image2 = cv2.imread(file_path2)
                        is_movement, score, details = detector.detect(image1, image2)
                        st.session_state.analysis_results = {
                            'type': 'image',
                            'is_movement': is_movement,
                            'score': score,
                            'details': details
                        }
                        # Clean up temporary files
                        try:
                            os.unlink(file_path1)
                            os.unlink(file_path2)
                        except:
                            pass
                        st.success("🎉 Analysis completed! Check the 'Results' tab for detailed results.")
            with col2:
                st.info("""
                **🖼️ Image Comparison Features:**
                - 🎯 Compare two images for movement detection
                - 🔍 Feature-based analysis using SIFT/ORB
                - 📊 Detailed movement scoring
                - 🔄 Camera movement detection between frames
                """)
        elif uploaded_file1 is not None or uploaded_file2 is not None:
            st.warning("⚠️ Please upload both images for comparison.")

    elif upload_type == "Video File":
        st.subheader("📹 Video Analysis")
        uploaded_file = st.file_uploader(
            "📁 Upload your video file",
            type=['mp4', 'avi', 'mov'],
            key="video_file",
            help="Upload a video file for movement analysis"
        )
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{uploaded_file.name.split(".")[-1]}') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                file_path = tmp_file.name
            video_col1, video_col2 = st.columns([1, 1])
            with video_col1:
                st.subheader("📹 Original Video")
                st.video(uploaded_file, width=350)
            with video_col2:
                st.subheader("🎯 Live Analysis")
                live_viz_placeholder = st.empty()
            col1, col2 = st.columns([1, 2])
            with col1:
                if st.button("🚀 Start Video Analysis", type="primary", use_container_width=True):
                    with st.spinner("🔍 Analyzing video..."):
                        if analysis_type in ["📹 Camera Only", "🔄 Both"]:
                            detector = CameraMovementDetector(
                                method=params['camera_method'],
                                threshold=params['threshold'],
                                min_match_count=params['min_match_count']
                            )
                            st.info("📹 Running camera movement analysis...")
                            camera_results = detector.analyze_video(
                                file_path, 
                                params['camera_max_frames'], 
                                params['frame_skip'],
                                enable_live_viz=params['enable_live_viz'],
                                live_viz_placeholder=live_viz_placeholder
                            )
                            st.success(f"✅ Camera movement analysis completed: {len(camera_results['movement_frames'])} movement frames detected")
                        else:
                            camera_results = None
                        if analysis_type in ["🎯 Object Only", "🔄 Both"]:
                            if params['object_method'] == "Lucas-Kanade":
                                st.info("🎯 Running Lucas-Kanade object movement analysis...")
                                analyzer = LucasKanadeAnalyzer(
                                    max_corners=params['max_corners'],
                                    quality_level=params['quality_level'],
                                    min_distance=params['min_distance']
                                )
                                object_results = analyzer.analyze_video(
                                    file_path, 
                                    params['object_max_frames'], 
                                    params['frame_skip'],
                                    enable_live_viz=params['enable_live_viz'],
                                    live_viz_placeholder=live_viz_placeholder
                                )
                                st.success(f"✅ Lucas-Kanade completed: {len(object_results['object_frames'])} object movement frames detected")
                            else:
                                st.info("🌊 Running Farneback object movement analysis...")
                                analyzer = FarnebackAnalyzer(
                                    object_threshold=params['object_threshold'],
                                    flow_threshold=params['flow_threshold']
                                )
                                object_results = analyzer.analyze_video(
                                    file_path, 
                                    params['object_max_frames'], 
                                    params['frame_skip'],
                                    enable_live_viz=params['enable_live_viz'],
                                    live_viz_placeholder=live_viz_placeholder
                                )
                                st.success(f"✅ Farneback completed: {len(object_results['object_frames'])} object movement frames detected")
                        else:
                            object_results = None
                        st.session_state.analysis_results = {
                            'type': 'video',
                            'movement_frames': camera_results['movement_frames'] if camera_results else [],
                            'movement_scores': camera_results['movement_scores'] if camera_results else [],
                            'details_list': camera_results['details_list'] if camera_results else [],
                            'frames': camera_results['frames'] if camera_results else [],
                            'total_frames': camera_results['total_frames'] if camera_results else 0,
                            'object_results': object_results,
                            'analysis_type': analysis_type
                        }
                        st.success("🎉 Analysis completed! Check the 'Results' tab for detailed results.")
            with col2:
                st.info("""
                **🎥 Video Analysis Features:**
                - 📹 Camera movement detection using SIFT/ORB
                - 🎯 Object movement detection using Lucas-Kanade/Farneback
                - 👁️ Real-time visualization
                - ⚡ Performance optimization with frame skipping
                """)

with tab2:
    display_results_tab()

with tab3:
    st.subheader("👩‍💻 About Developer")
    
    st.markdown("""
    **👩‍💻 Şeyma Ertuğrul**
    
    This application is developed by me: Şeyma 💖 Hi, I mean just hire me!
    """) 
    
    st.markdown("**🔗 Connect with me:**")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('GitHub: https://github.com/SeymaErtugrul')
        st.markdown('LinkedIn: https://www.linkedin.com/in/seyma-ertugrul-18b1aa199/')

with tab4:
    st.subheader("🎥 About Application")
    
    st.subheader("🎥 Movement Detection Demo")
    st.markdown("""
    This application demonstrates computer vision techniques for detecting movement in videos and images.
    
    **🎯 Features:**
    - **📹 Camera Movement Detection**: Uses SIFT/ORB algorithms to detect camera movement
    - **🎯 Object Movement Detection**: Uses Lucas-Kanade/Farneback optical flow for object movement
    - **👁️ Real-time Visualization**: See analysis results as they're processed
    - **⚡ Performance Optimization**: Frame skipping and memory management
    - **🔄 Multiple Analysis Types**: Choose camera-only, object-only, or both analyses
    
    **🔧 Technologies Used:**
    - OpenCV for computer vision
    - Streamlit for web interface
    - Plotly for data visualization
    - NumPy for numerical computations
    
    **📖 How to Use:**
    1. 📤 Upload a video or image file
    2. ⚙️ Configure analysis parameters in the sidebar
    3. 🚀 Click "Start Analysis"
    4. 📊 View results in the Results tab
    
    **💡 Tips:**
    - 📉 Lower thresholds for more sensitive detection
    - ⏭️ Use frame skipping for faster processing
    - 👁️ Enable live visualization for real-time feedback
    """)
