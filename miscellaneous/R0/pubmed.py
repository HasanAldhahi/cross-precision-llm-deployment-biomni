import os
import requests
import sys
import socket

# --- NEW PROXY CONFIGURATION (ADMIN RECOMMENDED) ---
# Using the hostname might allow internal DNS to route us to the correct,
# accessible internal proxy IP.
proxy_hostname = "www-cache.gwdg.de"
proxy_port = 3128
proxy_url = f"http://{proxy_hostname}:{proxy_port}"

proxies = {
   "http": proxy_url,
   "https": proxy_url,
}

print("=" * 60)
print("PubMed Connection Test Script v2")
print("=" * 60)
print(f"Node: {os.uname().nodename}")

# Let's see what IP this hostname resolves to from inside the cluster
try:
    proxy_ip = socket.gethostbyname(proxy_hostname)
    print(f"DNS Resolution: '{proxy_hostname}' resolves to '{proxy_ip}'")
except Exception as e:
    print(f"DNS Resolution FAILED for '{proxy_hostname}': {e}")

print(f"Using Proxy Configuration: {proxies}")

# --- PubMed API Request ---
search_term = "ovarian neoplasms"
base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
params = {
    "db": "pubmed",
    "term": search_term,
    "retmax": 5,
    "retmode": "json",
    "email": "your_email@example.com", # Remember to set this if you want
    "tool": "gwdg_connection_test"
}

print(f"\nAttempting to connect to PubMed API...")
try:
    # Reduced timeout to 10 seconds for faster failure feedback
    response = requests.get(base_url, params=params, proxies=proxies, timeout=10)
    response.raise_for_status()
    data = response.json()

    print("\n" + "="*60)
    print("✓ SUCCESS: Connection to PubMed was successful!")
    print("=" * 60)
    paper_ids = data.get("esearchresult", {}).get("idlist", [])
    print(f"Found {len(paper_ids)} paper IDs: {', '.join(paper_ids)}")

except requests.exceptions.ProxyError as e:
    print("\n" + "!"*60)
    print("✗ FAILURE: Proxy Error.")
    print("The node CANNOT reach the proxy server at this address.")
    print(f"Error Details: {e}")
    print("!"*60)
except Exception as e:
    print("\n" + "!"*60)
    print(f"✗ FAILURE: Other Error.")
    print(f"Error Details: {e}")
    print("!"*60)