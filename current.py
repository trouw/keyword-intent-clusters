import streamlit as st
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.oauth2 import credentials as google_credentials
from client import RestClient
import requests
import time
import datetime
import io
import base64
import pandas as pd
import matplotlib.pyplot as plt

# OAuth 2.0 client ID and scopes
CLIENT_ID = 'your-client-id'
CLIENT_SECRET = 'your-client-secret'
SCOPES = ['https://www.googleapis.com/auth/webmasters.readonly']

# GSC API Version & Service Name
GSC_API_VERSION = 'v3'
GSC_API_NAME = 'webmasters'

keyword_data_df = pd.DataFrame(columns=['Keyword', 'Clicks', 'Impressions', 'CTR', 'Position'])

# Global variables for OAuth2
CLIENT_ID = ''
CLIENT_SECRET = ''
PROJECT_ID = ''

# Function to convert the dataframe to a downloadable format
def to_csv_download_link(df, filename="data.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV File</a>'
    return href


# Function to query for keywords through GSC API and convert to DataFrame
@st.cache_data
# def query_keyword_data(access_token, site_url, start_date, end_date):
#     try:
#         # Explicitly specify the credentials when creating the service
#         credentials = google_credentials.Credentials(access_token)
#         gsc_service = build(GSC_API_NAME, GSC_API_VERSION, credentials=credentials)

#         # Convert date objects to strings
#         start_date_str = start_date.strftime('%Y-%m-%d')
#         end_date_str = end_date.strftime('%Y-%m-%d')

#         # Prepare the initial query
#         request = gsc_service.searchanalytics().query(
#             siteUrl=site_url,
#             body={
#                 'startDate': start_date_str,
#                 'endDate': end_date_str,
#                 'dimensions': ['query'],
#                 'rowLimit': 50  # Max row limit per request
#             }
#         )

#         keyword_data = []

#         # Function to recursively fetch all available data with pagination
#         def fetch_data(request):
#             response = request.execute()
#             if 'rows' in response:
#                 keyword_data.extend(response['rows'])

#             next_page_token = response.get('nextPageToken')
#             if next_page_token:
#                 request['startRow'] = response['rows'][len(response['rows']) - 1]['position'] + 1
#                 request['pageToken'] = next_page_token
#                 fetch_data(request)

#         fetch_data(request)

#         # Convert the keyword data to a DataFrame
#         if keyword_data:
#             df = pd.DataFrame(keyword_data)
#             df.columns = ['Keyword', 'Clicks', 'Impressions', 'CTR', 'Position']
#             return df

#     except Exception as e:
#         st.error(f"Error fetching keyword data: {str(e)}")

#     return None

# Function to preprocess data and apply filters
def preprocess_data(data, exclude_keywords=None, include_keywords=None, exclude_urls=None, include_urls=None, min_max_dict=None):
    if data is not None:
        # Determine the column names for keywords and URLs
        keyword_col = next((col for col in data.columns if col.lower() in ['keyword', 'query']), None)
        url_col = next((col for col in data.columns if col.lower() in ['url', 'page']), None)
        
        # Handle keyword filtering
        if exclude_keywords or include_keywords:
            if keyword_col:
                if exclude_keywords:
                    exclusion_mask = data[keyword_col].apply(lambda x: any(keyword.lower() in x.lower() for keyword in exclude_keywords))
                    data = data[~exclusion_mask]
                if include_keywords:
                    inclusion_mask = data[keyword_col].apply(lambda x: any(keyword.lower() in x.lower() for keyword in include_keywords))
                    data = data[inclusion_mask]
            else:
                print("No 'keyword' or 'query' column found")

        # Handle URL filtering
        if exclude_urls or include_urls:
            if url_col:
                if exclude_urls:
                    exclusion_mask = data[url_col].apply(lambda x: any(url.lower() in x.lower() for url in exclude_urls))
                    data = data[~exclusion_mask]
                if include_urls:
                    inclusion_mask = data[url_col].apply(lambda x: any(url.lower() in x.lower() for url in include_urls))
                    data = data[inclusion_mask]
            else:
                print("No 'url' or 'page' column found")

        # Handle value range filtering
        if min_max_dict:
            for col, (min_value, max_value) in min_max_dict.items():
                print(f"Filtering {col}: min_value = {min_value}, max_value = {max_value}")  # Debugging statement
                data = data[(data[col] >= min_value) & (data[col] <= max_value)]
                print(data)
    return data


# Function to query DataForSEO SERP for multiple keywords
def query_dataforseo_serp(username, password, keywords, search_engine="google", search_type="organic", language_code="en", location_code=2840):
    # Create a RestClient instance
    client = RestClient(username, password)
    all_data = []
    keyword_intents = []  # Create a list to store the intent for each keyword

    total_keywords = len(keywords)
    progress_bar = st.progress(0)  # Initialize progress bar

    for index, keyword in enumerate(keywords):
        
        # Defining SERP features 
        zero = ["answer_box","featured_snippet"]
        one = ["people_also_ask","scholarly_articles", "questions_and_answers", "short_videos"]
        one_half = ["google_posts", "knowledge_graph"]
        two = ["carousel", "top_stories", "video"]
        two_half = ["google_flights", "images", "related_searches", "people_also_search", "recipes", "find_results_on","found_on_the_web", "refine_products"]
        three = ["app", "multi_carousel", "math_solver", "visual_stories"]
        three_half = ["jobs"]
        four = ["events", "mention_carousel", "podcasts"]
        five = ["map", "twitter", "currency_box", "explore_brands"]
        six = ["local_pack", "hotels_pack", "top_sights"]
        seven_half = ["google_reviews", "stocks_box"]
        eight = ["paid", "popular_products", "local_services", "google_hotels",]
        nine = ["commercial_units"]
        ten = ["shopping"]
        
        keyword_intent = []
        # Update progress bar
        progress_bar.progress((index + 1) / total_keywords)
        post_data = dict()

        post_data[len(post_data)] = {
            "language_code": language_code,
            "location_code": location_code,
            "keyword": keyword,
            "calculate_rectangles": True
        }

        endpoint = f"/v3/serp/{search_engine}/{search_type}/live/advanced"
        
        response = client.post(endpoint, post_data)

        if response["status_code"] == 20000:
            # Extracting organic results
            all_results = response['tasks'][0]['result']

            #Determining Intent
            for i in response['tasks'][0]['result'][0]['item_types']:
                if i in zero:
                    keyword_intent.append(0)
                if i in one:
                    keyword_intent.append(1)
                if i in one_half:
                    keyword_intent.append(1.5)
                if i in two:
                    keyword_intent.append(2)
                if i in two_half:
                    keyword_intent.append(2.5)
                if i in three:
                    keyword_intent.append(3)
                if i in three_half:
                    keyword_intent.append(3.5)
                if i in four:
                    keyword_intent.append(4)
                if i in five:
                    keyword_intent.append(5)
                if i in six:
                    keyword_intent.append(6)
                if i in seven_half:
                    keyword_intent.append(7.5)
                if i in eight:
                    keyword_intent.append(8)
                if i in nine:
                    keyword_intent.append(9)
                if i in ten:
                    keyword_intent.append(10)
            
            intent_avg = (sum(keyword_intent)/len(keyword_intent))

            # Find the organic results & verify SERP features within the list
            organic_results = []
            for res in all_results:
                if res.get('type') == 'organic':
                    organic_results.extend(res['items'])


            # Limit to 15 results
            organic_results = organic_results[:15]
            
            # Create a list to store the extracted data for this keyword
            data_list = []

            # Iterate through the organic results and extract relevant information
            for result in organic_results:
                keyword = keyword
                intent_avg=intent_avg
                url = result.get('url')
                position = result.get('rank_absolute')
                title = result.get('title')
                description = result.get('description')
                
                # Append the extracted data to the list
                data_list.append([keyword, url, position, title, description, intent_avg])

            # Add the data for this keyword to the list of all data
            all_data.extend(data_list)
        
        else:
            print(f"Error for keyword '{keyword}': Code: {response['status_code']} Message: {response['status_message']}")

    # Create a DataFrame from the combined data for all keywords
    df = pd.DataFrame(all_data, columns=["Keyword", "URL", "Position", "Title", "Description", "Keyword Intent"])
    
    return df

def extract_info_from_serp(data):
    extracted_info = []

    # Define a recursive function to go through the nested structure
    def extract_from_item(item):
        if not isinstance(item, dict):
            return

        # Base information to extract
        info = {
            'type': item.get('type'),
            'title': item.get('title'),
            'url': item.get('url')
        }

        # Filter out entries without a title or URL
        if info['title'] or info['url']:
            extracted_info.append(info)

        # Check for nested items and recurse if found
        if 'items' in item:
            for subitem in item['items']:
                extract_from_item(subitem)

    # Iterate over the main data and start extraction
    for element in data:
        extract_from_item(element)

    return extracted_info

def serps_similarity(df):
    # Assuming df contains SERP data with 'keyword', 'URL' and 'Position' columns

    # Filter for Page 1 results
    df = df[df['Position'] <= 10]
    
    # Group by keyword and join URLs into a single string
    serp_strings = df.groupby('Keyword').apply(lambda group: ' '.join(map(str, group['URL'])))
    
    # Create a DataFrame to store similarity scores
    similarity_df = pd.DataFrame(index=serp_strings.index, columns=serp_strings.index)
    
    # Compare SERP similarity
    for keyword_a, serp_string_a in serp_strings.items():
        for keyword_b, serp_string_b in serp_strings.items():
            # Simple similarity measure - Jaccard similarity
            set_a = set(serp_string_a.split())
            set_b = set(serp_string_b.split())
            jaccard_similarity = len(set_a.intersection(set_b)) / len(set_a.union(set_b))
            similarity_df.loc[keyword_a, keyword_b] = jaccard_similarity

            # Melting the similarity matrix
            melted_df = similarity_df.reset_index().melt(id_vars='Keyword', var_name='Keyword_B', value_name='Similarity')
    
    return melted_df

def create_clusters(similarity_df, data_df):
    clusters = {}
    for index, row in similarity_df.iterrows():
        keyword_a = row['Keyword']
        keyword_b = row['Keyword_B']
        similarity_score = row['Similarity']
        
        if similarity_score >= 0.4:  # Assuming similarity is in range [0, 1]
            if keyword_a not in clusters:
                clusters[keyword_a] = [keyword_b]
            else:
                clusters[keyword_a].append(keyword_b)
    
    cluster_data = []
    for cluster, keywords in clusters.items():
        keyword_data = data_df[data_df['Keyword'].isin(keywords)]
        total_clicks = keyword_data['Clicks'].sum()
        total_impressions = keyword_data['Impressions'].sum()
        avg_intent = keyword_data['Keyword Intent'].mean()
        cluster_data.append([cluster, total_clicks, total_impressions, avg_intent])
    
    cluster_df = pd.DataFrame(cluster_data, columns=['Cluster', 'Total Clicks', 'Total Impressions', 'Avg. Keyword Intent'])
    return cluster_df

def create_bubble_chart(agg_data):
    # Check if 'size_metric' is in session state, otherwise set default to 'Clicks'
    if 'size_metric' not in st.session_state:
        st.session_state.size_metric = 'Clicks'

    st.session_state.size_metric = st.selectbox('Choose size metric', ['Total Clicks', 'Total Impressions'])

    sizes = agg_data[str(st.session_state.size_metric)]
    fig, ax = plt.subplots()
    ax.scatter([0] * len(agg_data), agg_data['Avg. Keyword Intent'], s=sizes, alpha=0.5)
    st.pyplot(fig)

# Streamlit app
def main():
    filtered_df = None
    cluster_df = None
    filtered_data = None
    st.title("Data Preprocessor")
    
    with st.expander("Upload Data"):
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file, index_col=0)  # Use the first column as the index
            st.write("Uploaded Data:")
            st.write(data)
    
    with st.expander("Filter Data"):
        # Keyword filtering
        keyword_filter_toggle = st.checkbox('Enable Keyword Filtering')
        exclude_keywords, include_keywords = None, None
        if keyword_filter_toggle:
            keyword_action = st.radio('Keyword Action', ['Include', 'Exclude'])
            keywords_input = st.text_input('Keywords (comma separated)')
            if keyword_action == 'Exclude':
                exclude_keywords = [kw.strip() for kw in keywords_input.split(",")]
            else:
                include_keywords = [kw.strip() for kw in keywords_input.split(",")]
        
        # URL filtering
        url_filter_toggle = st.checkbox('Enable URL Filtering')
        exclude_urls, include_urls = None, None
        if url_filter_toggle:
            url_action = st.radio('URL Action', ['Include', 'Exclude'], key='url_action')
            urls_input = st.text_input('URLs (comma separated)', key='urls_input')
            if url_action == 'Exclude':
                exclude_urls = [url.strip() for url in urls_input.split(",")]
            else:
                include_urls = [url.strip() for url in urls_input.split(",")]
        
        exclude_dict = {}
        min_max_dict = {}
        
        for col in data.columns:
            if col.lower() in ['keyword', 'url']:
                exclude_dict[col] = st.text_input(f"Exclude values from {col} (comma-separated):")
            elif pd.api.types.is_numeric_dtype(data[col]):
                min_value, max_value = st.slider(f"{col} Range", float(data[col].min()), float(data[col].max()), (float(data[col].min()), float(data[col].max())))
                min_max_dict[col] = (min_value, max_value)
        
        if st.button("Filter Data"):
            filtered_data = preprocess_data(data, exclude_keywords, include_keywords, exclude_urls, include_urls, min_max_dict)
            st.write("Filtered Data:")
            st.write(filtered_data)

            
            # Download link for filtered data
            csv = filtered_data.to_csv(index=False)  # Do not write index to CSV
            b64 = base64.b64encode(csv.encode()).decode()
            st.download_button(
                label="Download Filtered Data",
                data=b64,
                file_name="filtered_data.csv",
                mime="text/csv"
            )

    with st.expander("DataForSEO API Integration"):
        st.warning("Running this part of the script will cost money. Ensure you have enough funds in your DataForSEO account.")

        # Input fields for DataForSEO API credentials
        username = st.text_input("DataForSEO Username (Email):")
        password = st.text_input("DataForSEO Password:", type="password")
        client = RestClient(username, password)

        # Button to query DataForSEO SERP
        if st.button("Query DataForSEO SERP"):
            if not username or not password:
                st.error("Please provide DataForSEO credentials.")
            else:
                # Get the list of keywords from the GSC DataFrame
                if 'keyword_data_df_filtered' in st.session_state:
                    keywords_to_query = st.session_state.keyword_data_df_filtered["Keyword"].tolist()

                    # Run DataForSEO API query for multiple keywords
                    result_df = query_dataforseo_serp(username, password, keywords_to_query)
                    st.session_state['result_df'] = result_df

                    if result_df is not None:
                        # Display the DataFrame
                        st.write("Data from DataForSEO API:")
                        st.dataframe(result_df)

                        # You can further process and save this DataFrame as needed
                    else:
                        st.error("Error querying DataForSEO API.")
                else:
                    st.warning("No filtered keyword data available. Fetch and filter keyword data from GSC first.")

    with st.expander("SERP Similarity"):
        if 'result_df' in st.session_state:
            similarity_df = serps_similarity(st.session_state['result_df'])
            st.write("SERP Similarity Matrix:")
            st.dataframe(similarity_df.reset_index())  # Reset index to include 'Keyword' column
            
            if 'keyword_data_df_filtered' in st.session_state and 'result_df' in st.session_state:
                merged_df = pd.merge(st.session_state['keyword_data_df_filtered'], st.session_state['result_df'][['Keyword', 'Keyword Intent']], on='Keyword', how='inner')
                cluster_df = create_clusters(similarity_df, merged_df)

            # Adding a download button for the SERP similarity matrix
            csv1 = similarity_df.reset_index().to_csv(index=False)  # Reset index to include 'Keyword' column
            b64 = base64.b64encode(csv1.encode()).decode()
            st.download_button(
                label="Download SERP Similarity Matrix",
                data=b64,
                file_name="serp_similarity_matrix.csv",
                mime="text/csv"
            )
            st.dataframe(cluster_df)

    with st.expander("Viz"):
        if cluster_df is not None:
            create_bubble_chart(cluster_df)

if 'oauth2_expander_state' not in st.session_state:
    st.session_state.oauth2_expander_state = False

if __name__ == "__main__":
    main()

