import streamlit as st

def generate_content_locally(topic, output_format, tone):
    """
    Generates content locally based on user inputs without an API call.
    """
    base_report = f"This is a {tone} analysis on the topic of '{topic}'. The report is formatted as an {output_format}."
    
    if output_format == "LinkedIn Post":
        post = f"{base_report}\n\nKey trends and insights are highlighted below. #TechTrends #MarketAnalysis #{topic.replace(' ', '')}"
        return post
    elif output_format == "Executive Summary":
        summary = f"{base_report}\n\nSummary: The key findings indicate a positive outlook with significant growth potential. Further details are available in the full report."
        return summary
    elif output_format == "Email Briefing":
        briefing = f"Subject: Briefing on '{topic}'\n\nDear Team,\n\nHere is a brief summary of the latest developments concerning '{topic}'. The outlook is {tone}.\n\nBest regards,\nYour Assistant"
        return briefing
    elif output_format == "Key Talking Points":
        points = f"{base_report}\n\n1. Point One: First key finding.\n2. Point Two: Second key finding.\n3. Point Three: Third key finding."
        return points
    
    return base_report

# --- UI Layout ---
st.set_page_config(page_title="Hugging Face News Assistant", page_icon="🤗", layout="wide")

# --- Header ---
col1, col2 = st.columns([1, 6])
with col1:
    st.image("https://huggingface.co/front/assets/huggingface_logo.svg", width=100) # Hugging Face Logo
with col2:
    st.title("Hugging Face Insight Assistant")
    st.markdown("### Generate Professional Content Locally")

st.markdown("---")

# --- Sidebar for Controls ---
st.sidebar.header("📝 Content Controls")
output_format = st.sidebar.selectbox(
    "Select Output Format",
    ("LinkedIn Post", "Executive Summary", "Email Briefing", "Key Talking Points")
)
tone = st.sidebar.selectbox(
    "Select Tone of Voice",
    ("Professional", "Analytical", "Casual", "Optimistic", "Cautious")
)

# --- Main Content Area ---
st.subheader("Enter a topic to begin your analysis")
topic = st.text_input("Topic", placeholder="e.g., 'advances in quantum computing'")


if st.button("Generate Intelligence", type="primary", use_container_width=True):
    if topic:
        with st.spinner("🤖 Generating your content..."):
            generated_content = generate_content_locally(topic, output_format, tone)
            
            if generated_content:
                st.markdown("---")
                st.subheader(f"Your Generated {output_format}")
                
                # Custom CSS for a cleaner, modern look
                st.markdown(
                    """
                    <style>
                    .content-container {
                        border: 1px solid #e1e4e8;
                        border-radius: 12px;
                        padding: 25px;
                        background-color: #f6f8fa;
                        box-shadow: 0 0 10px rgba(0,0,0,0.1);
                        margin-bottom: 20px;
                        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                        font-size: 16px;
                        line-height: 1.6;
                        color: #24292e;
                    }
                    </style>
                    """, unsafe_allow_html=True
                )

                formatted_content = generated_content.replace('\n', '<br>')
                st.markdown(f'<div class="content-container">{formatted_content}</div>', unsafe_allow_html=True)
                
    else:
        st.warning("Please enter a topic to generate content.")
