"""
Topo-Flow Streamlit Dashboard - Multi-Rock Edition
Upload rock chunks and predict permeability with specialized models
"""

import streamlit as st
import os
import sys
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.inference import predict_single_chunk


# Cache model loading
@st.cache_resource
def load_model(model_name, model_type='standard'):
    """Load specific GNN model (standard or hybrid)"""
    import torch
    from src.model import TopoFlowGNN
    from src.model_hybrid import HybridPhysicsGNN
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if model_type == 'hybrid':
        # Load hybrid model (only available for MEC currently)
        model = HybridPhysicsGNN(dropout=0.1).to(device)
        model_path = 'models/best_model_hybrid.pth'
    else:
        # Load standard GNN
        model_paths = {
            'MEC': 'models/best_model.pth',
            'Synthetic': 'models/best_model_synthetic.pth',
            'ILS': 'models/best_model_ils.pth',
            'Estaillades': 'models/best_model_estaillades.pth',
            'Savonnières': 'models/best_model_savonnieres.pth'
        }
        model = TopoFlowGNN(dropout=0.1).to(device)
        model_path = model_paths.get(model_name, model_paths['MEC'])
    
    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(state_dict)
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Failed to load {model_name} {model_type} model: {e}")
        return None, device


# Page config
st.set_page_config(
    page_title="Topo-Flow: Multi-Rock AI",
    page_icon="🪨",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
    }
    .winner-badge {
        background: #10b981;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: bold;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🪨 Topo-Flow Multi-Rock</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">5-Dataset Validated AI for Permeability Prediction</p>',
    unsafe_allow_html=True
)

# Sidebar
with st.sidebar:
    st.header("🎯 Select Rock Type")
    
    rock_type = st.selectbox(
        "Choose your rock type",
        ['MEC Carbonate', 'Indiana Limestone (ILS)', 'Synthetic Blobs', 'Estaillades (Vuggy) 🏆', 'Savonnières (3-Phase) 🔥'],
        help="Select the rock type that matches your sample"
    )
    
    # Map display name to model name
    rock_mapping = {
        'MEC Carbonate': 'MEC',
        'Indiana Limestone (ILS)': 'ILS',
        'Synthetic Blobs': 'Synthetic',
        'Estaillades (Vuggy) 🏆': 'Estaillades',
        'Savonnières (3-Phase) 🔥': 'Savonnières'
    }
    selected_model = rock_mapping[rock_type]
    
    st.markdown("---")
    st.header("🔬 Model Type")
    
    # Hybrid only available for MEC currently
    if selected_model == 'MEC':
        model_type = st.radio(
            "Choose prediction mode",
            ['Standard GNN', 'Hybrid (Formula + GNN)'],
            help="Hybrid combines Kozeny-Carman with GNN residual learning"
        )
        use_hybrid = (model_type == 'Hybrid (Formula + GNN)')
        
        if use_hybrid:
            st.info("💡 Hybrid mode: Kozeny-Carman baseline + GNN topology correction")
    else:
        use_hybrid = False
        st.caption("Hybrid mode available for MEC only")
    
    st.markdown("---")
    st.header("📁 Upload Rock Chunk")
    
    uploaded_file = st.file_uploader(
        "Choose a .npy file",
        type=['npy'],
        help="Upload a 128³ voxel chunk (binary: 0=solid, 1=pore)"
    )
    
    st.markdown("---")
    st.subheader(f"ℹ️ About {rock_type}")
    
    rock_info = {
        'MEC Carbonate': {
            'desc': 'Middle Eastern carbonate with moderate heterogeneity',
            'porosity': '5-25%',
            'k_range': '10⁻¹⁶ to 10⁻¹³ m²',
            'performance': 'Baseline wins (Kozeny-Carman better)',
            'samples': 374
        },
        'Indiana Limestone (ILS)': {
            'desc': 'Well-connected grainstone',
            'porosity': '10-25%',
            'k_range': '10⁻¹² to 10⁻¹¹ m²',
            'performance': 'Baseline wins (porosity dominated)',
            'samples': 266
        },
        'Synthetic Blobs': {
            'desc': 'Controlled topology, variable blobiness',
            'porosity': '15-35%',
            'k_range': '10⁻¹⁷ to 10⁻¹⁴ m²',
            'performance': 'Baseline wins (uniform structure)',
            'samples': 200
        },
        'Estaillades (Vuggy) 🏆': {
            'desc': 'Complex vuggy carbonate - GNN WINS HERE!',
            'porosity': '10-15%',
            'k_range': '10⁻¹² to 10⁻⁹ m²',
            'performance': '🎉 GNN wins by 28.4%!',
            'samples': 200
        },
        'Savonnières (3-Phase) 🔥': {
            'desc': '3-phase vuggy carbonate - BEST GNN WIN!',
            'porosity': '6-50%',
            'k_range': '10⁻¹⁷ to 10⁻¹⁰ m²',
            'performance': '🔥 GNN wins by 46.2%! (Best result)',
            'samples': 191
        }
    }
    
    info = rock_info[rock_type]
    st.markdown(f"""
    **Description:** {info['desc']}
    
    **Characteristics:**
    - Porosity: {info['porosity']}
    - K Range: {info['k_range']}
    - Training Samples: {info['samples']}
    
    **Performance:**
    {info['performance']}
    """)
    
    st.markdown("---")
    st.caption("Built with ❤️ using 1,231 samples across 5 rock types")

# Main content
if uploaded_file is not None:
    temp_path = "temp_chunk.npy"
    
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())
    
    st.success(f"✅ Uploaded: {uploaded_file.name} → {rock_type}")
    
    # Load and show stats
    chunk = np.load(temp_path)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Shape", f"{chunk.shape[0]}×{chunk.shape[1]}×{chunk.shape[2]}")
    with col2:
        phi = np.sum(chunk) / chunk.size
        st.metric("Porosity", f"{phi:.3f}")
    with col3:
        st.metric("Pore Voxels", f"{np.sum(chunk):,}")
    
    st.markdown("---")
    
    # Run inference
    mode_label = "Hybrid" if use_hybrid else "Standard"
    with st.spinner(f'🧠 Running {selected_model} Model ({mode_label})...'):
        try:
            predicted_k, image_path, baseline_k = predict_single_chunk(
                temp_path,
                output_image='streamlit_output.png',
                use_hybrid=use_hybrid,
                rock_type=selected_model
            )
            
            st.success("✅ Prediction Complete!")
            
            # Display prediction
            if use_hybrid and baseline_k is not None:
                # Show both baseline and hybrid
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"""
                    <div style="background: #6366f1; padding: 1.5rem; border-radius: 10px; color: white; text-align: center;">
                        <div style="font-size: 0.9rem; margin-bottom: 0.5rem;">Kozeny-Carman Baseline</div>
                        <div style="font-size: 2rem; font-weight: bold;">{baseline_k:.4e} m²</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    improvement = ((baseline_k - predicted_k) / baseline_k) * 100
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 10px; color: white; text-align: center;">
                        <div style="font-size: 0.9rem; margin-bottom: 0.5rem;">Hybrid (Formula + GNN) 🔬</div>
                        <div style="font-size: 2rem; font-weight: bold;">{predicted_k:.4e} m²</div>
                        <div style="font-size: 0.9rem; margin-top: 0.5rem; color: #a5f3fc;">Improvement: {improvement:+.2f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.info(f"💡 The GNN learned a correction of {abs(improvement):.2f}% to the physics-based formula!")
            else:
                # Standard display
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">Predicted Permeability ({selected_model})</div>
                    <div class="metric-value">{predicted_k:.4e} m²</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Show if this is the winning model
            if selected_model == 'Estaillades':
                st.markdown('<div class="winner-badge">🏆 GNN SUPERIOR ON THIS ROCK TYPE!</div>', unsafe_allow_html=True)
                st.info("Estaillades vuggy carbonate: GNN beats Kozeny-Carman by 28.4% due to complex topology!")
            
            # Visualization
            if image_path and os.path.exists(image_path):
                st.subheader("🎨 3D Flow Visualization")
                st.image(image_path, caption="Pore Network Flow", use_container_width=True)
            
            # Details
            with st.expander("📊 Detailed Information"):
                st.markdown(f"""
                **Model:** {selected_model}
                - Architecture: Graph Attention Network (GAT)
                - Training Samples: {info['samples']}
                
                **Prediction:**
                - Log10(K): {np.log10(predicted_k):.4f}
                - Permeability: {predicted_k:.4e} m²
                
                **Expected Range:** {info['k_range']}
                
                **Research Finding:**
                {info['performance']}
                """)
            
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
        except Exception as e:
            st.error(f"❌ Error: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)

else:
    # Welcome screen
    st.info("👆 Select rock type and upload a .npy file to get started!")
    
    # Research findings
    st.subheader("🔬 Research Findings (1,040 Samples)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **✅ When GNN Wins:**
        - **Estaillades vuggy carbonate** 🏆
        - High heterogeneity (450x K variation)
        - Complex vug connectivity
        - **28.4% better than Kozeny-Carman**
        
        **Use GNN for:**
        - Vuggy carbonates
        - Disconnected pore regions
        - Multi-scale porosity
        """)
    
    with col2:
        st.markdown("""
        **✅ When Baseline Wins:**
        - **MEC, ILS, Synthetic** 
        - Uniform pore structure
        - Well-connected networks
        - **5-187x better than GNN**
        
        **Use Kozeny-Carman for:**
        - Homogeneous rocks
        - Grainstones
        - Synthetic media
        """)
    
    # Performance comparison
    st.markdown("---")
    st.subheader("📊 Model Performance Summary")
    
    import pandas as pd
    
    results_df = pd.DataFrame({
        'Rock Type': ['MEC Carbonate', 'Synthetic Blobs', 'Indiana Limestone', 'Estaillades Vuggy'],
        'Samples': [374, 200, 266, 200],
        'Baseline MSE': [0.0018, 0.2337, 0.0251, 0.1120],
        'GNN MSE': [0.3372, 1.2666, 0.3273, 0.0802],
        'Winner': ['Baseline (187x)', 'Baseline (5.4x)', 'Baseline (13x)', '🏆 GNN (28.4%)']
    })
    
    st.dataframe(results_df, use_container_width=True)
    
    st.success("💡 **Conclusion:** Topology matters for vuggy rocks, porosity dominates for uniform media")

# Footer
st.markdown("---")
st.caption("Topo-Flow v2.0 | 4-Dataset Validated GNN for Permeability Prediction")
