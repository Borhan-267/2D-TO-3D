import streamlit as st
import pandas as pd
import numpy as np
import io
import os
import zipfile
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors, rdMolTransforms
import py3Dmol
import streamlit.components.v1 as components

# ================= PAGE CONFIGURATION =================
st.set_page_config(
    page_title="IRC Chem3D Studio | Molecular Structure Converter",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= CUSTOM LIGHT THEME CSS (IRC BRANDING) =================
st.markdown("""
<style>
    /* Force Clean Light Background & Dark Readable Text */
    .stApp {
        background-color: #F8FAFC;
        color: #0F172A;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E2E8F0;
    }

    /* Hide default Streamlit clutter */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Top Brand Banner */
    .irc-header {
        background: linear-gradient(135deg, #FFFFFF 0%, #EFF6FF 100%);
        border: 1px solid #DBEAFE;
        border-left: 6px solid #0D47A1;
        padding: 1.3rem 1.8rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(13, 71, 161, 0.05);
        margin-bottom: 1.2rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .irc-title {
        color: #0D47A1;
        font-size: 1.9rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .irc-subtitle {
        color: #475569;
        font-size: 0.95rem;
        margin-top: 4px;
        font-weight: 500;
    }
    .irc-badge {
        background-color: #ECFDF5;
        color: #15803D;
        border: 1px solid #86EFAC;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.5px;
    }

    /* Pipeline Step Bar */
    .pipeline-bar {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        padding: 12px 16px;
        border-radius: 10px;
        font-size: 0.85rem;
        color: #475569;
        margin-bottom: 1.5rem;
        text-align: center;
        font-weight: 600;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
    }
    .pipeline-step {
        color: #0D47A1;
        font-weight: 700;
    }

    /* Metric Cards Styling */
    div[data-testid="metric-container"] {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-top: 4px solid #0D47A1;
        padding: 16px 20px;
        border-radius: 10px;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04);
    }
    div[data-testid="metric-container"] label {
        color: #64748B !important;
        font-weight: 600 !important;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #0F172A !important;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #FFFFFF;
        padding: 6px;
        border-radius: 10px;
        border: 1px solid #E2E8F0;
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        color: #475569;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #EFF6FF !important;
        color: #0D47A1 !important;
    }

    /* Primary Button Override */
    .stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #0D47A1 0%, #1976D2 100%);
        border: none;
        color: #FFFFFF;
        font-weight: 600;
        padding: 0.65rem 1.6rem;
        border-radius: 8px;
        box-shadow: 0 4px 10px rgba(13, 71, 161, 0.2);
        transition: all 0.2s ease;
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(90deg, #1565C0 0%, #2E7D32 100%);
        box-shadow: 0 6px 14px rgba(13, 71, 161, 0.28);
    }

    /* Custom Footer */
    .irc-footer {
        margin-top: 3.5rem;
        padding: 1.2rem 0;
        border-top: 1px solid #E2E8F0;
        text-align: center;
        color: #64748B;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# ================= REFERENCE TEST DATASET =================
TEST_DRUGS = [
    {"ID": "IRC-001", "Name": "Aspirin", "SMILES": "CC(=O)OC1=CC=CC=C1C(=O)O"},
    {"ID": "IRC-002", "Name": "Ibuprofen (Chiral)", "SMILES": "CC(C)CC1=CC=C(C=C1)[C@H](C)C(=O)O"},
    {"ID": "IRC-003", "Name": "Paracetamol", "SMILES": "CC(=O)NC1=CC=C(O)C=C1"},
    {"ID": "IRC-004", "Name": "Caffeine", "SMILES": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"},
    {"ID": "IRC-005", "Name": "Atorvastatin (Multi-Chiral)", "SMILES": "CC(C)C1=C(C(=C(N1CC[C@H](C[C@H](CC(=O)O)O)O)C2=CC=C(F)C=C2)C3=CC=CC=C3)C(=O)NC4=CC=CC=C4"},
    {"ID": "IRC-006", "Name": "Oseltamivir", "SMILES": "CCOC(=O)C1=C[C@@H](OC(CC)CC)[C@H](NC(C)=O)[C@@H](N)C1"},
    {"ID": "IRC-007", "Name": "Sildenafil", "SMILES": "CCCC1=NN(C2=C1N=C(NC2=O)C3=C(C=CC(=C3)S(=O)(=O)N4CCN(C)CC4)OCC)C"},
    {"ID": "IRC-008", "Name": "Omeprazole", "SMILES": "CC1=CN=C(C(=C1OC)C)CS(=O)C2=NC3=C(N2)C=C(C=C3)OC"},
    {"ID": "IRC-009", "Name": "Doxorubicin Analog", "SMILES": "COCCN[C@H]1C[C@H](O[C@H]2C[C@](O)(C(=O)CO)Cc3c(O)c4c(c(O)c32)C(=O)c2c(OC)cccc2C4=O)O[C@@H](C)[C@H]1O"},
    {"ID": "IRC-010", "Name": "Invalid_Structure_Test", "SMILES": "C1=CC=CC(C)(C)(C)(C)=C1"}
]

# ================= CORE CHEMINFORMATICS ENGINE =================
def validate_geometry_and_stereo(mol_2d, mol_3d, conf_id):
    """Validates stereochemistry preservation and covalent bond length reasonability."""
    Chem.AssignStereochemistryFrom3D(mol_3d, confId=conf_id)
    chiral_2d = Chem.FindMolChiralCenters(mol_2d, includeUnassigned=True)
    chiral_3d = Chem.FindMolChiralCenters(mol_3d, includeUnassigned=True)
    
    stereo_preserved = (chiral_2d == chiral_3d)
    
    conf = mol_3d.GetConformer(conf_id)
    unreasonable_bonds = 0
    for bond in mol_3d.GetBonds():
        dist = rdMolTransforms.GetBondLength(conf, bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        if dist < 0.7 or dist > 2.5:
            unreasonable_bonds += 1
            
    is_chemically_valid = stereo_preserved and (unreasonable_bonds == 0)
    
    messages = []
    if not stereo_preserved:
        messages.append(f"Chirality mismatch: 2D{chiral_2d} vs 3D{chiral_3d}")
    if unreasonable_bonds > 0:
        messages.append(f"{unreasonable_bonds} abnormal bond length(s) detected")
    if is_chemically_valid:
        messages.append("Verified (Stereo & Geometry Valid)")
        
    return is_chemically_valid, "; ".join(messages), len(chiral_2d)

def process_single_molecule(mol_id, name, smiles, num_confs=15, max_iters=500):
    """Executes 2D-to-3D embedding, force-field energy minimization, and validation."""
    result = {
        "ID": mol_id,
        "Name": name,
        "SMILES": smiles,
        "Status": "Failed",
        "Force_Field": "N/A",
        "Min_Energy_kcal_mol": None,
        "Formula": "N/A",
        "Mol_Weight": None,
        "Chiral_Centers": 0,
        "Validation_Report": "N/A",
        "Error_Reason": "",
        "SDF_Block": None
    }
    
    try:
        if not isinstance(smiles, str) or not smiles.strip():
            result["Error_Reason"] = "Empty or non-string SMILES input"
            return result

        mol_2d = Chem.MolFromSmiles(smiles.strip())
        if mol_2d is None:
            result["Error_Reason"] = "RDKit SMILES parsing error (Invalid syntax or valence)"
            return result

        result["Formula"] = rdMolDescriptors.CalcMolFormula(mol_2d)
        result["Mol_Weight"] = round(rdMolDescriptors.CalcExactMolWt(mol_2d), 2)

        mol_3d = Chem.AddHs(mol_2d)

        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        params.useSmallRingTorsions = True
        params.useMacrocycleTorsions = True
        params.enforceChirality = True
        
        conf_ids = AllChem.EmbedMultipleConfs(mol_3d, numConfs=num_confs, params=params)
        
        if len(conf_ids) == 0:
            params.useRandomCoords = True
            conf_ids = AllChem.EmbedMultipleConfs(mol_3d, numConfs=num_confs, params=params)
            if len(conf_ids) == 0:
                result["Error_Reason"] = "ETKDGv3 3D conformer embedding failed"
                return result

        energies = []
        ff_used = "MMFF94s"
        
        if AllChem.MMFFHasAllMoleculeParams(mol_3d):
            mmff_props = AllChem.MMFFGetMoleculeProperties(mol_3d, mmffVariant="MMFF94s")
            for cid in conf_ids:
                ff = AllChem.MMFFGetMoleculeForceField(mol_3d, mmff_props, confId=cid)
                if ff is not None:
                    converged = ff.Minimize(maxIts=max_iters)
                    energy = ff.CalcEnergy()
                    energies.append((cid, energy, converged))
        
        if not energies:
            ff_used = "UFF (Fallback)"
            for cid in conf_ids:
                ff = AllChem.UFFGetMoleculeForceField(mol_3d, confId=cid)
                if ff is not None:
                    converged = ff.Minimize(maxIts=max_iters)
                    energy = ff.CalcEnergy()
                    energies.append((cid, energy, converged))

        if not energies:
            result["Error_Reason"] = "Force field initialization failed (MMFF94s & UFF)"
            return result

        energies.sort(key=lambda x: x[1])
        best_cid, min_energy, _ = energies[0]

        is_valid, val_msg, chiral_count = validate_geometry_and_stereo(mol_2d, mol_3d, best_cid)
        
        mol_3d.SetProp("_Name", f"{mol_id}_{name}")
        mol_3d.SetProp("Organization", "Innovative Research Center (IRC)")
        mol_3d.SetProp("SMILES", smiles)
        mol_3d.SetProp("ForceField", ff_used)
        mol_3d.SetProp("Potential_Energy_kcal_mol", f"{min_energy:.4f}")
        mol_3d.SetProp("Validation_Status", val_msg)

        sdf_stream = io.StringIO()
        writer = Chem.SDWriter(sdf_stream)
        writer.write(mol_3d, confId=best_cid)
        writer.close()

        result["Status"] = "Success" if is_valid else "Warning"
        result["Force_Field"] = ff_used
        result["Min_Energy_kcal_mol"] = round(min_energy, 4)
        result["Chiral_Centers"] = chiral_count
        result["Validation_Report"] = val_msg
        result["SDF_Block"] = sdf_stream.getvalue()
        
        if not is_valid:
            result["Error_Reason"] = val_msg

        return result

    except Exception as e:
        result["Error_Reason"] = f"Runtime Exception: {str(e)}"
        return result

def render_3d_viewer(sdf_data, style="Stick & Sphere"):
    """Renders an interactive 3D molecular viewport with a crisp white background."""
    view = py3Dmol.view(width=720, height=430)
    view.addModel(sdf_data, "sdf")
    if style == "Stick & Sphere":
        view.setStyle({"stick": {"radius": 0.15}, "sphere": {"scale": 0.25}})
    elif style == "Stick Only":
        view.setStyle({"stick": {"radius": 0.18}})
    else:
        view.setStyle({"sphere": {"scale": 0.75}})
    view.setBackgroundColor("#FFFFFF")
    view.zoomTo()
    components.html(view._make_html(), height=440, width=730)

# ================= SIDEBAR BRANDING & PARAMETERS =================
with st.sidebar:
    if os.path.exists("irc_logo.png"):
        st.image("irc_logo.png", use_container_width=True)
    else:
        st.markdown("""
        <div style="text-align:center; padding: 14px; border: 2px solid #0D47A1; border-radius: 10px; background: #FFFFFF;">
            <span style="color:#0D47A1; font-size:26px; font-weight:900;">IRC</span><br>
            <span style="color:#2E7D32; font-size:11px; font-weight:700; letter-spacing:1px;">INNOVATIVE RESEARCH CENTER</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("⚙️ Engine Configuration")
    
    num_confs = st.slider(
        "Conformers per Molecule",
        min_value=1, max_value=50, value=15,
        help="Higher values increase the probability of finding the global energy minimum for flexible rings."
    )
    max_iters = st.number_input(
        "Max Minimization Iterations",
        min_value=100, max_value=2000, value=500, step=100,
        help="Maximum steps for MMFF94s / UFF force-field energy convergence."
    )
    
    st.markdown("---")
    st.markdown("""
    **Computational Standards:**
    - **Embedding:** ETKDGv3 (Torsion-angle preference)
    - **Primary FF:** MMFF94s (Static variant)
    - **Fallback FF:** Universal Force Field (UFF)
    - **Validation:** 3D CIP Chirality & Bond Sanity
    """)

# ================= MAIN HEADER BANNER =================
st.markdown("""
<div class="irc-header">
    <div>
        <h1 class="irc-title">IRC Chem3D Studio</h1>
        <div class="irc-subtitle">Enterprise 2D-to-3D Molecular Conformer Generation & Force-Field Optimization Pipeline</div>
    </div>
    <div class="irc-badge">INNOVATIVE RESEARCH CENTER</div>
</div>

<div class="pipeline-bar">
    <span class="pipeline-step">1. SMILES Input</span> &nbsp;➔&nbsp;
    <span class="pipeline-step">2. Explicit H Addition</span> &nbsp;➔&nbsp;
    <span class="pipeline-step">3. ETKDGv3 Embedding</span> &nbsp;➔&nbsp;
    <span class="pipeline-step">4. MMFF94s / UFF Optimization</span> &nbsp;➔&nbsp;
    <span class="pipeline-step">5. Stereo & Geometry Check</span> &nbsp;➔&nbsp;
    <span class="pipeline-step">6. SDF Export</span>
</div>
""", unsafe_allow_html=True)

# ================= INPUT SECTION =================
st.subheader("1. Molecular Input Source")
input_mode = st.radio(
    "Select Input Method:",
    ["Single SMILES Entry", "Bulk CSV Upload (High-Throughput)", "Benchmark Validation Suite (10 Reference Drugs)"],
    horizontal=True,
    label_visibility="collapsed"
)

input_df = None

if input_mode == "Single SMILES Entry":
    c1, c2 = st.columns([1, 2.5])
    with c1:
        mol_name = st.text_input("Compound Identifier / Name", value="IRC_Lead_01")
    with c2:
        single_smiles = st.text_input(
            "Canonical or Isomeric SMILES",
            value="COCCN[C@H]1C[C@H](O[C@H]2C[C@](O)(C(=O)CO)Cc3c(O)c4c(c(O)c32)C(=O)c2c(OC)cccc2C4=O)O[C@@H](C)[C@H]1O"
        )
    if single_smiles:
        input_df = pd.DataFrame([{"ID": "MOL-001", "Name": mol_name, "SMILES": single_smiles}])

elif input_mode == "Bulk CSV Upload (High-Throughput)":
    uploaded_file = st.file_uploader("Upload CSV Dataset (Must contain a column with SMILES strings)", type=["csv"])
    if uploaded_file is not None:
        df_uploaded = pd.read_csv(uploaded_file)
        st.caption(f"Loaded {len(df_uploaded):,} rows. Preview of first 5 entries:")
        st.dataframe(df_uploaded.head(), use_container_width=True)
        
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            smiles_col = st.selectbox("Map SMILES Column:", df_uploaded.columns)
        with col_s2:
            id_col = st.selectbox("Map Compound ID / Name Column:", ["Auto-Generate IDs"] + list(df_uploaded.columns))
        
        input_df = pd.DataFrame()
        input_df["SMILES"] = df_uploaded[smiles_col]
        if id_col == "Auto-Generate IDs":
            input_df["ID"] = [f"IRC-{i+1:05d}" for i in range(len(df_uploaded))]
            input_df["Name"] = input_df["ID"]
        else:
            input_df["ID"] = df_uploaded[id_col].astype(str)
            input_df["Name"] = df_uploaded[id_col].astype(str)

elif input_mode == "Benchmark Validation Suite (10 Reference Drugs)":
    st.info("Pre-loaded benchmark suite containing 9 FDA-approved/chiral drug structures and 1 intentionally invalid structure to verify error handling.")
    input_df = pd.DataFrame(TEST_DRUGS)
    st.dataframe(input_df, use_container_width=True)

# ================= EXECUTION & RESULTS =================
if input_df is not None:
    if st.button("🚀 Run 3D Conformer Generation & Optimization", type="primary"):
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        total = len(input_df)

        for idx, row in input_df.iterrows():
            status_text.markdown(f"**Processing (`{idx+1}/{total}`):** `{row['Name']}`...")
            res = process_single_molecule(row["ID"], row["Name"], row["SMILES"], num_confs=num_confs, max_iters=max_iters)
            results.append(res)
            progress_bar.progress((idx + 1) / total)

        status_text.empty()
        st.session_state["results_df"] = pd.DataFrame(results)

if "results_df" in st.session_state:
    res_df = st.session_state["results_df"]
    total = len(res_df)
    success_count = len(res_df[res_df["Status"] == "Success"])
    warning_count = len(res_df[res_df["Status"] == "Warning"])
    failed_count = len(res_df[res_df["Status"] == "Failed"])

    st.markdown("---")
    st.subheader("2. Optimization & Validation Summary")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Processed", f"{total:,}")
    m2.metric("Verified 3D Structures", f"{success_count:,}")
    m3.metric("Geometry Warnings", f"{warning_count:,}")
    m4.metric("Failed Structures", f"{failed_count:,}")

    tab_view, tab_report, tab_errors, tab_export = st.tabs([
        "🔬 Interactive 3D Viewport",
        "📊 Optimization Report",
        "❌ Error & Exception Log",
        "💾 Export SDF & Reports"
    ])

    valid_mols = res_df[res_df["SDF_Block"].notnull()]

    with tab_view:
        if len(valid_mols) > 0:
            v_col1, v_col2 = st.columns([1.2, 2.5])
            with v_col1:
                selected_label = st.selectbox(
                    "Select Compound to Inspect in 3D:",
                    options=valid_mols.apply(lambda r: f"{r['ID']} | {r['Name']}", axis=1).tolist()
                )
                selected_id = selected_label.split(" | ")[0]
                mol_row = valid_mols[valid_mols["ID"] == selected_id].iloc[0]
                
                render_style = st.radio("3D Representation Style:", ["Stick & Sphere", "Stick Only", "Space-Filling (CPK)"])
                
                st.markdown(f"""
                **Compound Properties:**
                - **Formula:** `{mol_row['Formula']}`
                - **Exact Mass:** `{mol_row['Mol_Weight']} Da`
                - **Force Field:** `{mol_row['Force_Field']}`
                - **Min Energy:** `{mol_row['Min_Energy_kcal_mol']} kcal/mol`
                - **Chiral Centers:** `{mol_row['Chiral_Centers']}`
                - **Status:** `{mol_row['Validation_Report']}`
                """)
            with v_col2:
                st.caption("Left-click + drag to rotate | Scroll to zoom | Right-click + drag to pan")
                render_3d_viewer(mol_row["SDF_Block"], style=render_style)
        else:
            st.warning("No valid 3D structures available to render.")

    with tab_report:
        display_cols = ["ID", "Name", "Formula", "Mol_Weight", "Status", "Force_Field", "Min_Energy_kcal_mol", "Chiral_Centers", "Validation_Report", "SMILES"]
        st.dataframe(res_df[display_cols], use_container_width=True)

    with tab_errors:
        error_df = res_df[res_df["Status"] != "Success"]
        if len(error_df) > 0:
            st.error(f"{len(error_df)} compound(s) encountered processing errors or geometry warnings.")
            st.dataframe(error_df[["ID", "Name", "SMILES", "Status", "Error_Reason"]], use_container_width=True)
            err_csv = error_df[["ID", "Name", "SMILES", "Status", "Error_Reason"]].to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Error Log (CSV)", data=err_csv, file_name="IRC_3D_Error_Report.csv", mime="text/csv")
        else:
            st.success("All compounds passed 3D generation, energy minimization, and stereochemical validation without errors.")

    with tab_export:
        if len(valid_mols) > 0:
            e1, e2, e3 = st.columns(3)
            combined_sdf = "".join(valid_mols["SDF_Block"].tolist())
            with e1:
                st.download_button(
                    label=f"📦 Download Combined SDF ({len(valid_mols)} Mols)",
                    data=combined_sdf.encode("utf-8"),
                    file_name="IRC_Combined_3D_Structures.sdf",
                    mime="chemical/x-mdl-sdfile",
                    type="primary",
                    use_container_width=True
                )
            
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                for _, row in valid_mols.iterrows():
                    safe_name = "".join([c if c.isalnum() or c in ("-", "_") else "_" for c in f"{row['ID']}_{row['Name']}"])
                    zf.writestr(f"{safe_name}.sdf", row["SDF_Block"])
            with e2:
                st.download_button(
                    label="🗂️ Download Individual SDFs (ZIP)",
                    data=zip_buffer.getvalue(),
                    file_name="IRC_Individual_3D_SDFs.zip",
                    mime="application/zip",
                    use_container_width=True
                )

            with e3:
                full_csv = res_df.drop(columns=["SDF_Block"]).to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📄 Download Full Report (CSV)",
                    data=full_csv,
                    file_name="IRC_Optimization_Report.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        else:
            st.warning("No valid SDF structures generated for export.")

# ================= FOOTER =================
st.markdown("""
<div class="irc-footer">
    © 2026 <strong>Innovative Research Center (IRC)</strong> — Computational Chemistry & Drug Discovery Division. All Rights Reserved.
</div>
""", unsafe_allow_html=True)