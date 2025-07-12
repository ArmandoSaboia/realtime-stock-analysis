import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time

# --- Enhanced Custom Styling ---
def apply_enhanced_styling():
    """Apply advanced custom CSS for a modern, attractive Streamlit app."""
    st.markdown("""
    <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Global Styles */
        .main .block-container {
            padding-top: 1rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }
        
        /* Custom Font */
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }
        
        /* Header Styles */
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 15px;
            margin-bottom: 2rem;
            text-align: center;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }
        
        .main-header h1 {
            color: white;
            font-size: 2.5rem;
            font-weight: 700;
            margin: 0;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .main-header p {
            color: rgba(255,255,255,0.9);
            font-size: 1.1rem;
            margin: 0.5rem 0 0 0;
        }
        
        /* Enhanced Metric Cards */
        .metric-card {
            background: white;
            border-radius: 15px;
            padding: 1.5rem;
            margin: 0.5rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            border: 1px solid rgba(0,0,0,0.05);
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }
        
        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        }
        
        .metric-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, #667eea, #764ba2);
        }
        
        .metric-title {
            font-size: 0.9rem;
            font-weight: 500;
            color: #6B7280;
            margin-bottom: 0.5rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .metric-value {
            font-size: 2.2rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }
        
        .metric-change {
            font-size: 0.9rem;
            font-weight: 500;
            padding: 0.25rem 0.5rem;
            border-radius: 20px;
            display: inline-block;
        }
        
        .positive { color: #10B981; background: rgba(16,185,129,0.1); }
        .negative { color: #EF4444; background: rgba(239,68,68,0.1); }
        .neutral { color: #6B7280; background: rgba(107,114,128,0.1); }
        
        /* Dashboard Grid */
        .dashboard-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1rem;
            margin: 2rem 0;
        }
        
        /* Status Badges */
        .status-badge {
            padding: 0.5rem 1rem;
            border-radius: 25px;
            font-size: 0.85rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            display: inline-block;
            margin: 0.25rem;
        }
        
        .status-connected { background: #10B981; color: white; }
        .status-disconnected { background: #EF4444; color: white; }
        .status-loading { background: #F59E0B; color: white; }
        
        /* Sidebar Enhancements */
        .sidebar .sidebar-content {
            background: linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%);
        }
        
        /* Alert Boxes */
        .alert {
            padding: 1rem;
            border-radius: 10px;
            margin: 1rem 0;
            border-left: 4px solid;
        }
        
        .alert-success {
            background: rgba(16,185,129,0.1);
            border-left-color: #10B981;
            color: #065f46;
        }
        
        .alert-warning {
            background: rgba(245,158,11,0.1);
            border-left-color: #F59E0B;
            color: #92400e;
        }
        
        .alert-error {
            background: rgba(239,68,68,0.1);
            border-left-color: #EF4444;
            color: #991b1b;
        }
        
        /* Loading Animation */
        .loading-spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 20px auto;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        /* Chart Container */
        .chart-container {
            background: white;
            border-radius: 15px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        }
        
        /* Navigation Pills */
        .nav-pill {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 0.75rem 1.5rem;
            border-radius: 25px;
            text-decoration: none;
            font-weight: 500;
            margin: 0.25rem;
            display: inline-block;
            transition: all 0.3s ease;
        }
        
        .nav-pill:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102,126,234,0.4);
        }
        
        /* Hide default Streamlit elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: #667eea;
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: #5a67d8;
        }
    </style>
    """, unsafe_allow_html=True)

# --- Enhanced Header Component ---
def create_dashboard_header():
    """Create an attractive dashboard header."""
    st.markdown("""
    <div class="main-header">
        <h1>📈 Real-Time Stock Market Analytics</h1>
        <p>Advanced AI-Powered Financial Analysis Dashboard</p>
    </div>
    """, unsafe_allow_html=True)

# --- Enhanced Metric Display ---
def create_metric_card(title, value, change=None, change_type="neutral", icon="📊"):
    """Create an enhanced metric card with better styling."""
    change_class = f"metric-change {change_type}"
    change_html = f'<div class="{change_class}">{change}</div>' if change else ""
    
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">{icon} {title}</div>
        <div class="metric-value">{value}</div>
        {change_html}
    </div>
    """, unsafe_allow_html=True)

# --- Market Overview Dashboard ---
def create_market_overview():
    """Create a comprehensive market overview with key metrics."""
    st.markdown("### 🌍 Market Overview")
    
    # Sample data - replace with real API data
    market_data = {
        "S&P 500": {"value": "4,567.89", "change": "+1.2%", "type": "positive"},
        "NASDAQ": {"value": "14,234.56", "change": "+0.8%", "type": "positive"},
        "DOW": {"value": "35,678.90", "change": "-0.3%", "type": "negative"},
        "VIX": {"value": "18.45", "change": "+2.1%", "type": "negative"},
        "Gold": {"value": "$1,987.65", "change": "+0.5%", "type": "positive"},
        "Oil": {"value": "$87.23", "change": "-1.2%", "type": "negative"}
    }
    
    # Create columns for metrics
    cols = st.columns(3)
    
    for i, (name, data) in enumerate(market_data.items()):
        with cols[i % 3]:
            create_metric_card(
                name, 
                data["value"], 
                data["change"], 
                data["type"],
                "📈" if data["type"] == "positive" else "📉"
            )

# --- Enhanced Stock Analysis ---
def create_stock_analysis_section():
    """Create an enhanced stock analysis section."""
    st.markdown("### 📊 Stock Analysis")
    
    # Stock selector with search
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        stock_symbol = st.text_input("🔍 Enter Stock Symbol", "AAPL", placeholder="e.g., AAPL, GOOGL, TSLA")
    
    with col2:
        timeframe = st.selectbox("⏰ Timeframe", ["1D", "1W", "1M", "3M", "6M", "1Y"])
    
    with col3:
        chart_type = st.selectbox("📈 Chart Type", ["Candlestick", "Line", "Area"])
    
    if st.button("🚀 Analyze Stock", type="primary"):
        with st.spinner("Analyzing stock data..."):
            # Simulate loading time
            time.sleep(2)
            
            # Create sample stock data
            dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
            np.random.seed(42)
            prices = 150 + np.cumsum(np.random.randn(len(dates)) * 0.5)
            
            # Create interactive chart
            fig = go.Figure()
            
            if chart_type == "Candlestick":
                # Generate OHLC data
                opens = prices + np.random.randn(len(prices)) * 0.5
                highs = prices + abs(np.random.randn(len(prices)) * 2)
                lows = prices - abs(np.random.randn(len(prices)) * 2)
                closes = prices
                
                fig.add_trace(go.Candlestick(
                    x=dates,
                    open=opens,
                    high=highs,
                    low=lows,
                    close=closes,
                    name=stock_symbol
                ))
            
            elif chart_type == "Line":
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=prices,
                    mode='lines',
                    name=stock_symbol,
                    line=dict(color='#667eea', width=3)
                ))
            
            else:  # Area
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=prices,
                    mode='lines',
                    name=stock_symbol,
                    fill='tonexty',
                    line=dict(color='#667eea', width=2)
                ))
            
            fig.update_layout(
                title=f"{stock_symbol} - {timeframe} Analysis",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                template="plotly_white",
                height=500,
                showlegend=True,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display key metrics
            st.markdown("#### 📈 Key Metrics")
            
            cols = st.columns(4)
            with cols[0]:
                create_metric_card("Current Price", f"${prices[-1]:.2f}", "+2.5%", "positive", "💰")
            with cols[1]:
                create_metric_card("Day High", f"${prices[-1]*1.02:.2f}", "↗️", "positive", "📈")
            with cols[2]:
                create_metric_card("Day Low", f"${prices[-1]*0.98:.2f}", "↘️", "negative", "📉")
            with cols[3]:
                create_metric_card("Volume", "1.2M", "+15%", "positive", "📊")

# --- Portfolio Section ---
def create_portfolio_section():
    """Create a portfolio overview section."""
    st.markdown("### 💼 Portfolio Overview")
    
    # Portfolio data
    portfolio_data = {
        'Symbol': ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN'],
        'Shares': [50, 25, 75, 30, 20],
        'Price': [185.50, 2750.30, 378.85, 245.67, 3180.75],
        'Value': [9275, 68757.5, 28413.75, 7370.1, 63615],
        'Change': [2.5, -1.2, 1.8, -3.4, 0.9]
    }
    
    df = pd.DataFrame(portfolio_data)
    df['Total Value'] = df['Shares'] * df['Price']
    
    # Portfolio summary
    total_value = df['Total Value'].sum()
    total_change = df['Change'].mean()
    
    cols = st.columns(3)
    with cols[0]:
        create_metric_card("Total Value", f"${total_value:,.2f}", f"{total_change:+.1f}%", "positive" if total_change > 0 else "negative", "💰")
    with cols[1]:
        create_metric_card("Day P&L", f"${total_value*0.012:,.2f}", "+1.2%", "positive", "📈")
    with cols[2]:
        create_metric_card("Total P&L", f"${total_value*0.078:,.2f}", "+7.8%", "positive", "🚀")
    
    # Portfolio allocation chart
    fig = px.pie(df, values='Total Value', names='Symbol', 
                 title="Portfolio Allocation",
                 color_discrete_sequence=px.colors.qualitative.Set3)
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

# --- News and Alerts Section ---
def create_news_alerts_section():
    """Create a news and alerts section."""
    st.markdown("### 📰 Market News & Alerts")
    
    # Sample news data
    news_items = [
        {
            "title": "📈 Tech Stocks Rally on AI Optimism",
            "time": "2 hours ago",
            "summary": "Major technology companies see significant gains as AI developments continue to drive investor confidence.",
            "impact": "positive"
        },
        {
            "title": "🏦 Fed Hints at Rate Stability",
            "time": "4 hours ago", 
            "summary": "Federal Reserve officials suggest interest rates may remain stable in upcoming meetings.",
            "impact": "neutral"
        },
        {
            "title": "⚡ Energy Sector Faces Headwinds",
            "time": "6 hours ago",
            "summary": "Oil prices decline amid global supply concerns and demand uncertainty.",
            "impact": "negative"
        }
    ]
    
    for news in news_items:
        impact_class = f"alert-{'success' if news['impact'] == 'positive' else 'warning' if news['impact'] == 'neutral' else 'error'}"
        st.markdown(f"""
        <div class="alert {impact_class}">
            <strong>{news['title']}</strong><br>
            <small>{news['time']}</small><br>
            {news['summary']}
        </div>
        """, unsafe_allow_html=True)

# --- Enhanced Main Dashboard ---
def create_enhanced_dashboard():
    """Create the main enhanced dashboard."""
    # Apply custom styling
    apply_enhanced_styling()
    
    # Create header
    create_dashboard_header()
    
    # Connection status
    col1, col2 = st.columns([3, 1])
    with col2:
        st.markdown('<div class="status-badge status-connected">🟢 APIs Connected</div>', unsafe_allow_html=True)
    
    # Market overview
    create_market_overview()
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Stock Analysis", "💼 Portfolio", "📰 News & Alerts", "🤖 AI Insights"])
    
    with tab1:
        create_stock_analysis_section()
    
    with tab2:
        create_portfolio_section()
    
    with tab3:
        create_news_alerts_section()
    
    with tab4:
        st.markdown("### 🤖 AI-Powered Insights")
        st.info("🚀 **AI Analysis**: Market sentiment is bullish with strong momentum in tech sectors. Consider diversifying positions and monitoring volatility indicators.")
        
        # AI recommendations
        st.markdown("#### 💡 AI Recommendations")
        recommendations = [
            "🎯 **AAPL**: Strong buy signal based on technical indicators",
            "⚠️ **TSLA**: Hold position, watch for breakthrough above $250",
            "📈 **GOOGL**: Accumulate on dips, strong fundamentals",
            "🔍 **MSFT**: Monitor cloud revenue growth in next earnings"
        ]
        
        for rec in recommendations:
            st.markdown(f"- {rec}")

# --- Sidebar Enhancement ---
def create_enhanced_sidebar():
    """Create an enhanced sidebar with better organization."""
    st.sidebar.markdown("### 🔧 Settings")
    
    # API Configuration
    with st.sidebar.expander("🔑 API Configuration", expanded=True):
        twelvedata_key = st.text_input("Twelve Data API Key", type="password", placeholder="Enter your API key")
        groq_key = st.text_input("Groq API Key", type="password", placeholder="Enter your API key")
        
        if st.button("🔌 Connect APIs"):
            if twelvedata_key and groq_key:
                st.success("✅ Connected successfully!")
            else:
                st.error("❌ Please enter both API keys")
    
    # Trading Preferences
    with st.sidebar.expander("⚙️ Trading Preferences"):
        risk_tolerance = st.select_slider("Risk Tolerance", options=["Conservative", "Moderate", "Aggressive"])
        investment_horizon = st.selectbox("Investment Horizon", ["Short-term", "Medium-term", "Long-term"])
        preferred_sectors = st.multiselect("Preferred Sectors", ["Technology", "Healthcare", "Finance", "Energy", "Consumer"])
    
    # Alerts Configuration
    with st.sidebar.expander("🔔 Alerts & Notifications"):
        price_alerts = st.checkbox("Price Movement Alerts")
        news_alerts = st.checkbox("Breaking News Alerts")
        portfolio_alerts = st.checkbox("Portfolio Performance Alerts")
    
    # Quick Actions
    st.sidebar.markdown("### ⚡ Quick Actions")
    if st.sidebar.button("🔄 Refresh Data"):
        st.rerun()
    
    if st.sidebar.button("💾 Export Portfolio"):
        st.sidebar.success("Portfolio exported!")
    
    if st.sidebar.button("📊 Generate Report"):
        st.sidebar.success("Report generated!")

# --- Main App Function ---
def main():
    """Main application function."""
    st.set_page_config(
        page_title="Stock Market Analytics Dashboard",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Create enhanced sidebar
    create_enhanced_sidebar()
    
    # Create main dashboard
    create_enhanced_dashboard()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6B7280; font-size: 0.9rem;">
        📈 <strong>Stock Market Analytics Dashboard</strong> | Built with Python, Streamlit, Twelve Data, Plotly & Grok | 
        © 2024 Your Company Name
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
