import streamlit as st
import pandas as pd
import numpy as np

# --- APP SETUP ---
st.set_page_config(page_title="Ultimate QARP Growth Screener", layout="wide")
st.title("🏛️ Ultimate QARP Stock Screener & Growth Analyzer")
st.markdown("**Buffett, Greenblatt & Peter Lynch Logic (Quality + Reasonable Price + High Growth)**")

st.sidebar.header("📁 Data Source Settings")
st.sidebar.info("Screener.in ya apni CSV file upload karein jisme Sales aur Profit growth ke data bhi hon.")

uploaded_file = st.sidebar.file_uploader("Upload Market Data (CSV/Excel)", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        # File read karna
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
            
        st.success(f"✅ Successfully loaded {len(df)} stocks from data file!")
        
        # Columns clean up (Spaces hatana)
        df.columns = df.columns.str.strip()
        
        # Numeric conversion for all key financial columns
        numeric_cols = ['CMP', 'P/E', 'ROCE', 'Debt to equity', 'EPS', 'Sales growth 3Years', 'Profit growth 3Years']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Agar user ki file mein column name thode alag hain toh unhe handle karna
        # (Jaise 'Sales growth 3Years' ya 'Sales Growth 3Years')
        sales_col = next((c for c in df.columns if 'sales' in c.lower() and 'growth' in c.lower()), None)
        profit_col = next((c for c in df.columns if 'profit' in c.lower() and 'growth' in c.lower()), None)
        
        st.write("### ⚙️ Applying Strict QARP + Growth Filters...")
        
        if 'ROCE' in df.columns and 'P/E' in df.columns:
            
            # QARP + Growth Strict Filtering Logic
            # Rule: ROCE >= 15, Low Debt < 0.5, P/E < 40, aur Sales/Profit Growth >= 10%
            conditions = (
                (df['ROCE'] >= 15) & 
                (df['Debt to equity'] < 0.5) & 
                (df['EPS'] > 0) & 
                (df['P/E'] > 0) & 
                (df['P/E'] < 40)
            )
            
            # Agar growth columns mil gaye hain toh unhe bhi condition mein jodo
            if sales_col:
                conditions = conditions & (df[sales_col] >= 10)
            if profit_col:
                conditions = conditions & (df[profit_col] >= 10)
                
            filtered_df = df[conditions].copy()
            
            # Ranking based on Growth + ROCE + Reasonable Valuation (Magic Formula style)
            filtered_df['Rank_ROCE'] = filtered_df['ROCE'].rank(ascending=False)
            filtered_df['Rank_PE'] = filtered_df['P/E'].rank(ascending=True)
            if profit_col:
                filtered_df['Rank_Profit'] = filtered_df[profit_col].rank(ascending=False)
                filtered_df['Master_Score'] = filtered_df['Rank_ROCE'] + filtered_df['Rank_PE'] + filtered_df['Rank_Profit']
            else:
                filtered_df['Master_Score'] = filtered_df['Rank_ROCE'] + filtered_df['Rank_PE']
                
            top_qarp = filtered_df.sort_values('Master_Score').head(25)
            
            st.write(f"### 🏆 Top {len(top_qarp)} High-Growth QARP Wealth Compounders Found")
            
            # Display columns selection
            show_cols = ['Name', 'CMP', 'P/E', 'ROCE', 'Debt to equity', 'EPS']
            if sales_col: show_cols.append(sales_col)
            if profit_col: show_cols.append(profit_col)
            
            display_cols = [c for c in show_cols if c in top_qarp.columns]
            
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
    st.warning("👈 Left side menu se apni market data CSV file upload karein jisme Sales aur Profit growth bhi ho!")
    
    # Sample format guide with growth columns
    with st.expander("ℹ️ Dekhein ki CSV file ka format kaisa hona chahiye (Sales & Profit Growth ke sath)"):
        st.write("Aapki CSV file mein yeh columns hone chahiye:")
        sample_data = pd.DataFrame({
            'Name': ['TCS', 'HDFC Bank', 'ITC'],
            'CMP': [3800, 1600, 450],
            'P/E': [28.5, 18.2, 26.4],
            'ROCE': [45.2, 19.8, 32.1],
            'Debt to equity': [0.02, 0.35, 0.05],
            'EPS': [135, 88, 17],
            'Sales growth 3Years': [14.2, 16.5, 12.0],
            'Profit growth 3Years': [15.8, 18.2, 14.5]
        })
        st.dataframe(sample_data)
