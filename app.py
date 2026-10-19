import streamlit as st
import pandas as pd
import numpy as np

# --- APP SETUP ---
st.set_page_config(page_title="Ultimate QARP Screener", layout="wide")
st.title("🏛️ Ultimate QARP Stock Screener & Analyzer")
st.markdown("**Buffett & Greenblatt QARP Logic (Quality at a Reasonable Price)**")

st.sidebar.header("📁 Data Source Settings")
st.sidebar.info("Screener.in ya kisi bhi source ki CSV file yahan upload karein taaki 1000+ stocks ka instant analysis ho sake.")

uploaded_file = st.sidebar.file_uploader("Upload Market Data (CSV/Excel)", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        # File read karna
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
            
        st.success(f"✅ Successfully loaded {len(df)} stocks from data file!")
        
        st.write("### ⚙️ Applying Strict QARP Master Logic...")
        
        # Columns clean up (Spaces hatana)
        df.columns = df.columns.str.strip()
        
        # Columns matching check
        required_cols = ['Name', 'CMP', 'P/E', 'ROCE', 'Debt to equity', 'EPS']
        
        # Agar user ki file mein alag naam hain toh columns mapping kar dena
        # Yahan hum standard QARP filters chalayenge
        if 'ROCE' in df.columns and 'P/E' in df.columns:
            
            # Numeric conversion
            for col in ['CMP', 'P/E', 'ROCE', 'Debt to equity', 'EPS']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # QARP Strict Filtering Logic
            filtered_df = df[
                (df['ROCE'] >= 15) & 
                (df['Debt to equity'] < 0.5) & 
                (df['EPS'] > 0) & 
                (df['P/E'] > 0) & 
                (df['P/E'] < 40)
            ].copy()
            
            # Greenblatt Magic Formula Style Ranking (Sasta + Achha)
            filtered_df['Rank_ROCE'] = filtered_df['ROCE'].rank(ascending=False)
            filtered_df['Rank_PE'] = filtered_df['P/E'].rank(ascending=True)
            filtered_df['Master_Score'] = filtered_df['Rank_ROCE'] + filtered_df['Rank_PE']
            
            top_qarp = filtered_df.sort_values('Master_Score').head(25)
            
            st.write(f"### 🏆 Top {len(top_qarp)} Wealth Compounding Stocks Found")
            
            display_cols = [c for c in ['Name', 'CMP', 'P/E', 'ROCE', 'Debt to equity', 'EPS'] if c in top_qarp.columns]
            
            st.dataframe(
                top_qarp[display_cols].style.background_gradient(subset=['ROCE'], cmap='Greens'), 
                use_container_width=True
            )
            st.balloons()
            
            st.write("#### 📊 Complete Processed Market Database:")
            st.dataframe(df, use_container_width=True)
            
        else:
            st.error("❌ Aapki file mein zaroori columns ('ROCE', 'P/E') nahi mile. Kripya sahi format wali file upload karein.")
            
    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
else:
    st.warning("👈 Left side menu se apni market data CSV file upload karein. Isse 1000+ companies ka data ek second mein process ho jayega!")
    
    # Sample format guide
    with st.expander("ℹ️ Dekhein ki CSV file ka format kaisa hona chahiye"):
        st.write("Aapki CSV file mein kam se kam yeh columns hone chahiye:")
        sample_data = pd.DataFrame({
            'Name': ['TCS', 'HDFC Bank', 'ITC'],
            'CMP': [3800, 1600, 450],
            'P/E': [28.5, 18.2, 26.4],
            'ROCE': [45.2, 19.8, 32.1],
            'Debt to equity': [0.02, 0.35, 0.05],
            'EPS': [135, 88, 17]
        })
        st.dataframe(sample_data)
