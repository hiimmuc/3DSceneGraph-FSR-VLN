"""HoloAgent — Motion Robot Assistant

Streamlit UI with:
- Left panel (80%): 2×2 grid — Scene Graph Viewer, Query Result Viewer,
  Robot Camera Stream, Placeholder
- Right panel (20%): Chat interface with text + voice input, TTS output

Configuration is loaded from config/holoagent_app/holoagent_app.yaml and the
active profile (config/holoagent_app/profiles/<HOLOAGENT_PROFILE>.yaml).

Launch:
    cd fsr_vln
    streamlit run application/holoagent_app.py
    HOLOAGENT_PROFILE=lab streamlit run application/holoagent_app.py
"""

import logging
import os
import sys

# Ensure the fsr_vln package root is on the path when run directly
_HERE = os.path.dirname(os.path.abspath(__file__))
_FSR_ROOT = os.path.dirname(_HERE)
if _FSR_ROOT not in sys.path:
    sys.path.insert(0, _FSR_ROOT)

import streamlit as st
from application.audio_utils import (
    is_piper_available,
    is_whisper_available,
    synthesize_speech,
    transcribe_audio,
)
from application.ros2_utils import (
    get_latest_frame,
    is_ros2_available,
    start_camera_subscriber,
)
from memory.hmsg.utils.llm_utils import MotionAgent, publish_navigation_goal
from omegaconf import OmegaConf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config loading (Hydra-style YAML, without the @hydra.main decorator so it
# is compatible with Streamlit's own argv handling)
# ---------------------------------------------------------------------------


@st.cache_resource
def _load_config():
    """Load base config merged with the active profile."""
    cfg_dir = os.path.join(_FSR_ROOT, "config", "holoagent_app")
    base_path = os.path.join(cfg_dir, "holoagent_app.yaml")
    profile_name = os.environ.get("HOLOAGENT_PROFILE", "custom")
    profile_path = os.path.join(cfg_dir, "profiles", f"{profile_name}.yaml")

    base = OmegaConf.load(base_path)
    # Strip the top-level `defaults` key — it's a Hydra directive, not data
    if "defaults" in base:
        base = OmegaConf.masked_copy(base, [k for k in base if k != "defaults"])

    if os.path.isfile(profile_path):
        profile = OmegaConf.load(profile_path)
        cfg = OmegaConf.merge(base, profile)
    else:
        logger.warning(
            "Profile '%s' not found at %s — using base config.", profile_name, profile_path
        )
        cfg = base

    return cfg


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="HoloAgent — Motion",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Load config once (cached)
_cfg = _load_config()

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------


def _init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "nav_state" not in st.session_state:
        st.session_state.nav_state = {}
    if "query_result_figure" not in st.session_state:
        st.session_state.query_result_figure = None
    if "camera_started" not in st.session_state:
        st.session_state.camera_started = False


_init_session_state()

# ---------------------------------------------------------------------------
# Cached heavy resources
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading scene graph…")
def _load_graph(graph_path: str):
    """Load the Graph object once and cache it across reruns."""
    if not graph_path or not os.path.isdir(graph_path):
        logger.warning(
            "graph_path '%s' is not a valid directory — scene graph disabled.", graph_path
        )
        return None

    try:
        from memory.hmsg.graph.graph import Graph

        # Graph.__init__ always calls _load_clip_model() (even in query-only mode),
        # so we must include the `models` section. The `pipeline` key is intentionally
        # omitted — its absence triggers query-only mode inside Graph.__init__.
        graph_cfg = OmegaConf.create(
            {
                "main": OmegaConf.to_container(_cfg.main, resolve=True),
                "models": {
                    "clip": {
                        "type": "ViT-L/14",
                        "checkpoint": "checkpoints/open_clip_pytorch_model.bin",
                    },
                    "sam": {
                        "type": "vit_h",
                        "checkpoint": "checkpoints/sam_vit_h_4b8939.pth",
                        "points_per_side": 12,
                        "pred_iou_thresh": 0.88,
                        "points_per_batch": 144,
                        "stability_score_thresh": 0.95,
                        "crop_n_layers": 0,
                        "min_mask_region_area": 100,
                    },
                },
            }
        )
        graph = Graph(graph_cfg)
        graph.load_hmsg_graph(graph_path)
        # Assign human-readable room names (e.g. "living room", "kitchen")
        # using the room_types list from config, exactly as visualize_query_graph does.
        room_types = list(_cfg.main.room_types) if _cfg.main.get("room_types") else []
        graph.generate_room_names(
            generate_method=_cfg.main.get("room_generation_method", "label"),
            default_room_types=room_types,
        )
        if _cfg.main.get("room_names_human_assign"):
            graph.set_room_names(room_names=list(_cfg.main.room_names_human_assign))
        return graph
    except Exception as e:
        logger.error("Failed to load Graph: %s", e)
        return None


@st.cache_resource(show_spinner="Loading Motion AI…")
def _load_agent(_graph):
    """Create MotionAgent once."""
    return MotionAgent(scene_graph=_graph)


@st.cache_resource(show_spinner="Rendering scene graph…")
def _render_scene_graph_cached(graph_path: str):
    """Render scene graph as an interactive Plotly figure (cached resource)."""
    from application.render_utils import render_full_scene_graph_plotly
    return render_full_scene_graph_plotly(graph_path)


# ---------------------------------------------------------------------------
# Sidebar — status indicators
# ---------------------------------------------------------------------------

_graph_path = str(_cfg.main.graph_path)

with st.sidebar:
    st.title("🤖 HoloAgent")
    st.markdown("**System Status**")

    graph = _load_graph(_graph_path)
    agent = _load_agent(graph)

    if graph is not None:
        st.success("✅ Scene graph loaded")
    else:
        st.warning(
            "⚠️ Scene graph not loaded\n\n"
            "Set `main.graph_path` in `config/holoagent_app/profiles/custom.yaml`"
        )

    if is_ros2_available():
        st.success("✅ ROS2 available")
        if not st.session_state.camera_started:
            start_camera_subscriber(str(_cfg.main.camera_topic))
            st.session_state.camera_started = True
    else:
        st.warning("⚠️ ROS2 not available")

    if is_whisper_available():
        st.success("✅ STT (faster-whisper)")
    else:
        st.warning("⚠️ STT not available\n\n`pip install faster-whisper`")

    _piper_model = str(_cfg.audio.piper_voice_model)
    if is_piper_available() and _piper_model:
        st.success("✅ TTS (piper)")
    else:
        st.warning(
            "⚠️ TTS not configured\n\n"
            "Set `audio.piper_voice_model` in `config/holoagent_app/profiles/custom.yaml`"
        )

    st.divider()
    if st.button("🔄 Refresh Scene Graph"):
        _render_scene_graph_cached.clear()
        st.rerun()

# ---------------------------------------------------------------------------
# Main layout: left / right
# ---------------------------------------------------------------------------

_left_ratio = int(_cfg.ui.left_panel_ratio)
_right_ratio = int(_cfg.ui.right_panel_ratio)
_camera_refresh_ms = int(_cfg.ui.camera_refresh_interval_ms)
_top_k = int(_cfg.main.top_k)

col_left, col_right = st.columns([_left_ratio, _right_ratio])

# ===========================================================================
# LEFT PANEL — Visualization (2×2 grid)
# ===========================================================================

with col_left:
    st.subheader("Visualization", divider="blue")
    top_row = st.columns(2)
    bot_row = st.columns(2)

    # -----------------------------------------------------------------------
    # TOP-LEFT: Full scene graph viewer
    # -----------------------------------------------------------------------
    with top_row[0]:
        st.markdown("**🗺️ Scene Graph — Pre-scanned Environment**")
        if os.path.isdir(_graph_path):
            fig = _render_scene_graph_cached(_graph_path)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True, key="scene_graph_chart")
            else:
                st.info("Scene graph rendering failed. Check logs.")
        else:
            st.info("Set `main.graph_path` in\n" "`config/holoagent_app/profiles/custom.yaml`")

    # -----------------------------------------------------------------------
    # TOP-RIGHT: Query result viewer
    # -----------------------------------------------------------------------
    with top_row[1]:
        st.markdown("**🔍 Query Scene Graph — Top-K Results**")
        if st.session_state.query_result_figure is not None:
            st.plotly_chart(
                st.session_state.query_result_figure,
                use_container_width=True,
                key="query_result_chart",
            )
        else:
            st.info("Results will appear here after a navigation query.")

    # -----------------------------------------------------------------------
    # BOTTOM-LEFT: Robot camera stream
    # -----------------------------------------------------------------------
    with bot_row[0]:
        st.markdown("**📷 Robot Camera — Live Stream**")
        frame = get_latest_frame()
        if frame is not None:
            import numpy as np

            rgb_frame = frame[:, :, ::-1]
            st.image(rgb_frame, width="stretch")
        else:
            if is_ros2_available():
                st.info(f"Waiting for camera frames on `{_cfg.main.camera_topic}`…")
            else:
                st.info("ROS2 not available — camera stream disabled.")

        try:
            from streamlit_autorefresh import st_autorefresh

            st_autorefresh(interval=_camera_refresh_ms, key="camera_refresh")
        except ImportError:
            pass

    # -----------------------------------------------------------------------
    # BOTTOM-RIGHT: Placeholder
    # -----------------------------------------------------------------------
    with bot_row[1]:
        st.markdown("**🔲 Placeholder**")
        st.info("Reserved for future features.")

# ===========================================================================
# RIGHT PANEL — Chat interface
# ===========================================================================

with col_right:
    st.subheader("Motion", divider="orange")
    st.caption("Your robot assistant 🤖")

    # --- Conversation history ---
    chat_container = st.container(height=550)
    with chat_container:
        for msg in st.session_state.messages:
            role = msg["role"]
            display_name = "Motion" if role == "assistant" else "You"
            avatar = "🤖" if role == "assistant" else "🧑"
            with st.chat_message(role, avatar=avatar):
                st.markdown(f"**{display_name}:** {msg['content']}")

    # Play TTS audio after rerun (rendered outside the spinner context)
    if st.session_state.get("_tts_bytes"):
        st.audio(st.session_state.pop("_tts_bytes"), format="audio/wav", autoplay=True)

    # --- Voice input ---
    audio_input_text = None
    if is_whisper_available():
        try:
            from audio_recorder_streamlit import audio_recorder

            audio_bytes = audio_recorder(
                text="",
                recording_color="#e74c3c",
                neutral_color="#6c757d",
                icon_size="2x",
                key="voice_recorder",
            )
            if audio_bytes:
                _lang = str(_cfg.audio.whisper_language) or None
                with st.spinner("Transcribing…"):
                    audio_input_text = transcribe_audio(
                        audio_bytes,
                        language=_lang,
                        model_size=str(_cfg.audio.whisper_model_size),
                        device=str(_cfg.audio.whisper_device),
                        compute_type=str(_cfg.audio.whisper_compute_type),
                    )
                if audio_input_text:
                    st.caption(f"🎤 Heard: *{audio_input_text}*")
        except ImportError:
            pass

    # --- Text input ---
    text_input = st.chat_input("Talk to Motion…", key="chat_input")

    user_message = text_input or audio_input_text

    if user_message:
        st.session_state.messages.append({"role": "user", "content": user_message})

        with st.spinner("Motion is thinking…"):
            response, action = agent.process_message(
                user_message,
                st.session_state.messages[:-1],
                st.session_state.nav_state,
            )

        st.session_state.messages.append({"role": "assistant", "content": response})

        if action:
            print(
                f"[Navigation] Target confirmed: '{action['name']}' | "
                f"x={action['x']:.4f}  y={action['y']:.4f}  z={action['z']:.4f}",
                flush=True,
            )
            published = publish_navigation_goal(action)
            status_icon = "✅" if published else "⚠️"
            st.toast(
                f"{status_icon} Navigation goal sent: **{action['name']}** "
                f"({action['x']:.2f}, {action['y']:.2f}, {action['z']:.2f})",
                icon="🚀",
            )

            if graph is not None:
                try:
                    from copy import deepcopy

                    import numpy as np
                    import open3d as o3d
                    from application.render_utils import render_query_result_plotly

                    _OBJ_COLORS = [
                        [1.0, 0.15, 0.15],
                        [0.15, 0.85, 0.15],
                        [0.15, 0.45, 1.0],
                        [1.0, 0.60, 0.10],
                        [0.80, 0.10, 0.80],
                        [0.10, 0.85, 0.85],
                    ]
                    _floor, rooms, objects, _ = graph.query_hierarchy(action["name"], top_k=_top_k)
                    room_pcd = o3d.geometry.PointCloud()
                    seen = set()
                    for room in rooms:
                        if room.room_id not in seen:
                            seen.add(room.room_id)
                            room_pcd = room_pcd + deepcopy(room.pcd)

                    obj_pcds, spheres, obj_names = [], [], []
                    for i, obj in enumerate(objects):
                        pcd = deepcopy(obj.pcd)
                        pcd.paint_uniform_color(_OBJ_COLORS[i % len(_OBJ_COLORS)])
                        obj_pcds.append(pcd)
                        center = np.array(obj.pcd.get_center())
                        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.15)
                        sphere.translate(center)
                        sphere.paint_uniform_color(_OBJ_COLORS[i % len(_OBJ_COLORS)])
                        spheres.append(sphere)
                        obj_names.append(getattr(obj, "name", f"Object {i + 1}"))

                    query_fig = render_query_result_plotly(
                        room_pcd,
                        obj_pcds,
                        spheres,
                        obj_names=obj_names,
                    )
                    if query_fig is not None:
                        st.session_state.query_result_figure = query_fig
                except Exception as e:
                    logger.warning("Query result rendering failed: %s", e)

        # TTS — only if piper voice model is configured
        if is_piper_available() and _piper_model:
            tts_bytes = synthesize_speech(
                response,
                voice_model=_piper_model,
                use_cuda=bool(_cfg.audio.piper_use_cuda),
            )
            if tts_bytes:
                st.session_state["_tts_bytes"] = tts_bytes

        st.rerun()
