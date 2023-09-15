

import streamlit as st

# CSS to style the horizontal navigation bar
css = """
<style>
.navbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background-color: #333;
    color: white;
    
    margin: auto;
    width: 1400px;
}

.navbar img {
    width: 150px;
    height: auto;
}

.navbar-text {
    font-size: 24px;
    text-align: center;
    flex-grow: 1;
}
</style>
"""

# Create the Streamlit app
def main():
    st.set_page_config(
        page_title="Horizontal Navbar Example",
        page_icon="📑",
        layout="wide",
    )

    # Display the CSS for styling
    st.markdown(css, unsafe_allow_html=True)

    # st.markdown('<div class="navbar"><img src="https://www.thephoenixgroup.com/media/yy5fzboy/phoenix-logo.svg" alt="Image 1"><div class="navbar-text">Knowledge repository</div><img src="https://www.tcs.com/content/dam/global-tcs/en/images/home/dark-theme.svg" alt="Image 2"></div>', unsafe_allow_html=True)
    st.markdown('<div class="navbar"><img src="https://jobs.accaglobal.com/getasset/c64f085b-be72-43cf-87aa-71e08f04a47a/" alt="Image 1"><div class="navbar-text">Knowledge Repository</div><img src="https://www.tcs.com/content/dam/global-tcs/en/images/home/dark-theme.svg" alt="Image 2"></div>', unsafe_allow_html=True)


    
if __name__ == "__main__":
    main()
