mkdir -p ~/.streamlit/

echo "
[general]
email = \"edward_lam@vfc.com\"
" > ~/.streamlit/credentials.toml
echo "
[server]
headless = true
enableCORS=false
port = $PORT
" > ~/.streamlit/config.toml
