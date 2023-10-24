from client import RestClient
from io import BytesIO 
from itertools import combinations
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import base64
import concurrent.futures
import requests

# Function to preprocess data and apply filters
def preprocess_data(data, exclude_keywords=None, include_keywords=None, exclude_urls=None, include_urls=None, min_max_dict=None):
    if data is not None:
        # Determine the column names for keywords and URLs
        keyword_col = next((col for col in data.columns if col.lower() in ['keyword', 'query']), None)
        url_col = next((col for col in data.columns if col.lower() in ['URL', 'page']), None)
        
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
                st.warning("No 'url' or 'page' column found")

        # Handle value range filtering
        if min_max_dict:
            for col, (min_value, max_value) in min_max_dict.items():
                data = data[(data[col] >= min_value) & (data[col] <= max_value)]

    return data

# Function to query DataForSEO SERP for multiple keywords

def query_dataforseo_serp(username, password, keywords, search_engine="google", search_type="organic", language_code="en", location_code=2840):
    # Defining SERP features
    zero = ["shopping"]
    one = ["commercial_units"]
    two = ["paid", "popular_products", "local_services", "google_hotels"]
    two_half = ["google_reviews", "stocks_box"]
    four = ["local_pack", "hotels_pack", "top_sights", "google_flights"]
    five = ["map", "twitter", "currency_box", "explore_brands"]
    six = ["events", "mention_carousel", "podcasts"]
    six_half = ["jobs"]
    seven = ["app", "multi_carousel", "math_solver", "visual_stories"]
    seven_half = ["images", "related_searches", "people_also_search", "recipes", "find_results_on", "found_on_the_web", "refine_products"]
    eight = ["carousel", "top_stories", "video"]
    eight_half = ["google_posts", "knowledge_graph"]
    nine = ["people_also_ask", "scholarly_articles", "questions_and_answers", "short_videos"]
    ten = ["answer_box", "featured_snippet"]

    # Initialize DataForSEO client with your credentials
    client = RestClient(username, password)

    # Prepare task parameters for multiple keywords
    task_params = [
        {
            "language_code": language_code,
            "location_code": location_code,
            "keyword": keyword
        }
        for keyword in keywords
    ]

    # Send a single API request to create tasks for all keywords
    endpoint = "/v3/serp/google/organic/task_post"
    response = client.post(endpoint, task_params)

    

    if response["status_code"] == 20000:
        # Check if tasks are ready
        response_ready = client.get("/v3/serp/google/organic/tasks_ready")
        if response_ready["status_code"] == 20000:
            results = []
            control = True
            while control == True:
                response_ready = client.get("/v3/serp/google/organic/tasks_ready")
                if len(response_ready['tasks']) == 0:
                    control = False
                for task in response_ready['tasks']:
                    if (task['result'] and (len(task['result']) > 0)):
                        for resultTaskInfo in task['result']:
                            if resultTaskInfo['endpoint_advanced']:
                                result = client.get(resultTaskInfo['endpoint_advanced'])
                                results.append(result)
        st.write(len(results))
        all_data = []
        for serp in results:
            keyword = serp['tasks'][0]['data']['keyword']
            keyword_results = serp['tasks'][0]['result'][0]

            keyword_intent = []
            # Update progress bar
            # progress_bar.progress((index + 1) / total_keywords)

            # Extract SERP features
            for i in keyword_results['item_types']:
                if i in zero:
                    keyword_intent.append(0)
                if i in one:
                    keyword_intent.append(1)
                if i in two:
                    keyword_intent.append(2)
                if i in two_half:
                    keyword_intent.append(2.5)
                if i in four:
                    keyword_intent.append(4)
                if i in five:
                    keyword_intent.append(5)
                if i in six:
                    keyword_intent.append(6)
                if i in six_half:
                    keyword_intent.append(6.5)
                if i in seven:
                    keyword_intent.append(7)
                if i in seven_half:
                    keyword_intent.append(7.5)
                if i in eight:
                    keyword_intent.append(8)
                if i in eight_half:
                    keyword_intent.append(8.5)
                if i in nine:
                    keyword_intent.append(9)
                if i in ten:
                    keyword_intent.append(10)

            if len(keyword_intent) != 0:
                intent_avg = (sum(keyword_intent) / len(keyword_intent))
            else:
                intent_avg = 0

            # Find the organic results & verify SERP features within the list
            organic_results = []
            for res in keyword_results['items']:
                if res.get('type') == 'organic':
                    organic_results.append(res)

            # Limit to 15 results
            organic_results = organic_results[:15]

            # Iterate through the organic results and extract relevant information
            for result in organic_results:
                url = result.get('url')
                position = result.get('rank_absolute')
                title = result.get('title')
                description = result.get('description')
                st.write()

                all_data.append([keyword, url, position, title, description, intent_avg])

        st.write(all_data)
        if len(response_ready['tasks']) == 0  and all_data:
            st.write(all_data)
            df = pd.DataFrame(all_data, columns=["Keyword", "URL", "Position", "Title", "Description", "Keyword Intent"])
            df
            return df
        
        else:
            print("No data available.")
            return None
    #     # Now 'results' contains the data for each completed task that you can process further.
    #     for x in results:
    #         # Do something with each result
    #         print(x)
    # else:
    #     print("Error getting completed tasks. Code: %d Message: %s" % (response_ready["status_code"], response_ready["status_message"]))


    
    # if response["status_code"] == 20000:
    #     all_data = []
    #     total_keywords = len(keywords)
    #     progress_bar = st.progress(0)  # Initialize progress bar

    #     # Iterate through the response to extract data for each keyword
    #     for index, keyword in enumerate(keywords):
    #         keyword_results = response['tasks'][index]['result'][0]

    #         keyword_intent = []
    #         # Update progress bar
    #         progress_bar.progress((index + 1) / total_keywords)

    #         # Extract SERP features
    #         for i in keyword_results['item_types']:
    #             if i in zero:
    #                 keyword_intent.append(0)
    #             if i in one:
    #                 keyword_intent.append(1)
    #             if i in two:
    #                 keyword_intent.append(2)
    #             if i in two_half:
    #                 keyword_intent.append(2.5)
    #             if i in four:
    #                 keyword_intent.append(4)
    #             if i in five:
    #                 keyword_intent.append(5)
    #             if i in six:
    #                 keyword_intent.append(6)
    #             if i in six_half:
    #                 keyword_intent.append(6.5)
    #             if i in seven:
    #                 keyword_intent.append(7)
    #             if i in seven_half:
    #                 keyword_intent.append(7.5)
    #             if i in eight:
    #                 keyword_intent.append(8)
    #             if i in eight_half:
    #                 keyword_intent.append(8.5)
    #             if i in nine:
    #                 keyword_intent.append(9)
    #             if i in ten:
    #                 keyword_intent.append(10)

    #         if len(keyword_intent) != 0:
    #             intent_avg = (sum(keyword_intent) / len(keyword_intent))
    #         else:
    #             intent_avg = 0

    #         # Find the organic results & verify SERP features within the list
    #         organic_results = []
    #         for res in keyword_results['items']:
    #             if res.get('type') == 'organic':
    #                 organic_results.append(res)

    #         # Limit to 15 results
    #         organic_results = organic_results[:15]

    #         # Create a list to store the extracted data for this keyword
    #         data_list = []

    #         # Iterate through the organic results and extract relevant information
    #         for result in organic_results:
    #             url = result.get('url')
    #             position = result.get('rank_absolute')
    #             title = result.get('title')
    #             description = result.get('description')

    # #             # Append the extracted data to the list
    # #             data_list.append([keyword, url, position, title, description, intent_avg])

    # #         # Add the data for this keyword to the list of all data
    # #         all_data.extend(data_list)

    # #     # Create a DataFrame from the combined data for all keywords
    # #     df = pd.DataFrame(all_data, columns=["Keyword", "URL", "Position", "Title", "Description", "Keyword Intent"])

    #     # Retrieve the list of completed tasks

    # # return df
    # # else:
    # #     print(f"Error creating tasks. Code: {response['status_code']} Message: {response['status_message']}")
    # #     return None
def jaccard_similarity(set1, set2):  #serp_sim dependency
    set1 = set(set1)
    set2 = set(set2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0.0

def remove_subsets(cluster_df): #serp_sim dependency
    clusters = []

    # Iterate through each row in the DataFrame
    for _, row in cluster_df.iterrows():
        cluster_added = False  # Flag to track if the row was added to an existing cluster

        # Check if the row's keywords are subsets of any existing cluster
        for i, existing_cluster in enumerate(clusters):
            if any(keyword in existing_cluster for keyword in [row['Keyword_A'], row['Keyword_B']]):
                # Add the row's keywords to the existing cluster
                clusters[i].add(row['Keyword_A'])
                clusters[i].add(row['Keyword_B'])
                cluster_added = True
                break

        if not cluster_added:
            # If the row's keywords are not subsets of any existing cluster, create a new cluster
            clusters.append({row['Keyword_A'], row['Keyword_B']})

    # Filter out duplicate keywords within clusters
    filtered_clusters = [set(unique_keywords) for cluster in clusters for unique_keywords in [list(cluster)]]

    return filtered_clusters

def serps_similarity(data):
    # Group by keyword and get lists of URLs
    url_lists = data.groupby('Keyword')['URL'].apply(list).reset_index()

    keywords = url_lists['Keyword'].tolist()
    url_lists = url_lists['URL'].tolist()

    # Initialize an empty matrix
    matrix = pd.DataFrame(0, index=keywords, columns=keywords)

    similar_keyword_pairs = []
    # Calculate URL similarity for all pairs of keywords
    for i, j in combinations(range(len(keywords)), 2):
        set1 = url_lists[i]
        set2 = url_lists[j]

        if set1 is not None and set2 is not None:
            common_urls = len(set(set1) & set(set2))  # Count common URLs
            matrix.loc[keywords[i], keywords[j]] = common_urls
            matrix.loc[keywords[j], keywords[i]] = common_urls

        if common_urls >= 4:
                similar_keyword_pairs.append({
                    'Keyword_A': keywords[i],
                    'Keyword_B': keywords[j],
                    'Similar_URLs': common_urls
                })

    # Create a DataFrame from the extracted similar keyword pairs
    similar_pairs_df = pd.DataFrame(similar_keyword_pairs)

    # Remove subsets from clusters
    clusters = remove_subsets(similar_pairs_df)

    st.session_state['Clusters'] = clusters

    return clusters

def aggregate_clusters(cluster_data, keyword_df):
    # Create an empty DataFrame to store cluster-level data
    cluster_agg_df = pd.DataFrame()

    # Iterate through the list of clusters
    for cluster_keywords in cluster_data:
        # Determine the cluster name (keyword with highest Search Volume or Impressions)
        cluster_name_keyword = None

        if 'clicks' in keyword_df.columns and 'impressions' in keyword_df.columns:
            # If both Clicks and Impressions columns are present in keyword data, choose the keyword with the highest Impressions
            cluster_keyword_data = keyword_df[keyword_df['Keyword'].isin(cluster_keywords)]
            cluster_name_keyword = keyword_df.loc[cluster_keyword_data['impressions'].idxmax()]['Keyword']
            
            # Aggregate data within the cluster
            cluster_agg = {
                'Cluster Name': cluster_name_keyword,
                'Cluster': ', '.join(cluster_keywords),
                'Keyword Intent': cluster_keyword_data['Keyword Intent'].mean(),
                'Clicks': cluster_keyword_data['clicks'].sum(),
                'Impressions': cluster_keyword_data['impressions'].sum()
            }

            # Append the cluster-level data to the cluster_agg_df DataFrame
            cluster_agg_df = cluster_agg_df.append(cluster_agg, ignore_index=True)

        elif 'Search Volume' in keyword_df.columns:
            # If only Search Volume is present in keyword data, choose the keyword with the highest Search Volume
            cluster_keyword_data = keyword_df[keyword_df['Keyword'].isin(cluster_keywords)]
            cluster_name_keyword = keyword_df.loc[cluster_keyword_data['Search Volume'].idxmax()]['Keyword']
            
            # Aggregate data within the cluster
            cluster_agg = {
                'Cluster Name': cluster_name_keyword,
                'Cluster': ', '.join(cluster_keywords),
                'Keyword Intent': cluster_keyword_data['Keyword Intent'].mean(),
                'Search Volume': cluster_keyword_data['Search Volume'].sum()
            }

            # Append the cluster-level data to the cluster_agg_df DataFrame
            cluster_agg_df = cluster_agg_df.append(cluster_agg, ignore_index=True)

    return cluster_agg_df

def create_bubble_chart(agg_data, x_limit, y_limit, font_size):
    available_metrics = []
    if 'Clicks' in agg_data.columns:
        available_metrics.append('Clicks')
    if 'Impressions' in agg_data.columns:
        available_metrics.append('Impressions')
    if 'Search Volume' in agg_data.columns:
        available_metrics.append('Search Volume')

    # Check if 'size_metric' is in session state, otherwise set default
    if 'size_metric' not in st.session_state or st.session_state.size_metric not in available_metrics:
        st.session_state.size_metric = available_metrics[0]
    
    # Slider for font size
    font_size = st.slider('Font Size', min_value=2, max_value=16, value=font_size)

    # Slider for x-axis limit
    x_limit = st.slider('X-axis Limit', min_value=5, max_value=20, value=x_limit)

    # Slider for y-axis limit
    y_limit_range = st.slider('Y-axis Limit', min_value=0, max_value=20, value=(0, y_limit))

    # Extract the lower and upper bounds of the y-axis limit from the tuple
    y_limit_lower, y_limit_upper = y_limit_range

    circle_size = st.slider('Circle Size', min_value=1, max_value=10)

    # Now create the bubble chart
    sizes = (agg_data[str(st.session_state.size_metric)]/circle_size)
    fig, ax = plt.subplots()

    # Set the y-axis limits based on the range selected
    ax.set_ylim(y_limit_lower, y_limit_upper)

    # Set the x-axis limits
    ax.set_xlim(-x_limit, x_limit)

    # Generate numeric y-axis values based on keyword intent
    y_values = agg_data['Keyword Intent']

    # Scatter plot with y-axis as keyword intent values and x-axis as 0 (centered)
    ax.scatter([0] * len(agg_data), y_values, s=sizes, alpha=0.5)

    # Add cluster names to the center of bubbles with the specified font size
    for i, label in enumerate(agg_data['Cluster Name']):
        ax.text(0, y_values.iloc[i], label, ha='center', va='center', fontsize=font_size)

    # Add a static background image using the specified x and y limits
    image_url = 'https://raw.githubusercontent.com/trouw/keyword-intent-clusters/main/marketing-funnel.PNG'
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    ax.imshow(image, extent=[-10, 10, 0, 10], alpha=0.5)

    ax.set_aspect('auto')

    # Show the chart
    st.pyplot(fig)


# Streamlit app
def main():
    filtered_data = None
    st.title("Keyword Intent Cluster Tool")
    
    with st.expander("Upload Data"):
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file, header=0, encoding='utf-8')  # Try UTF-8 first
            except UnicodeDecodeError:
                try:
                    data = pd.read_csv(uploaded_file, header=0, encoding='iso-8859-1')  # Try ISO-8859-1 next
                except UnicodeDecodeError:
                    data = pd.read_csv(uploaded_file, header=0, encoding='windows-1252') 
            st.write("Uploaded Data:")
            st.dataframe(data)
            st.write(f'Row count before filtering: {len(data)}')

            # Identify the column name for keywords
            keyword_col_name = next((col for col in data.columns if col.lower() in ['keyword', 'query']), None)
            if keyword_col_name != 'Keyword':
                data.rename(columns={keyword_col_name: 'Keyword'}, inplace=True)
            elif keyword_col_name == 'Keyword':
                None
            else:
                st.warning("No column for keywords found. Expected column name to be 'keyword' or 'query'.")
            
            st.session_state['data'] = data
    
    if 'data' in st.session_state: 
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
            
            min_max_dict = {}
            
            for col in data.columns:
                if  pd.api.types.is_numeric_dtype(data[col]):
                    min_value, max_value = st.slider(f"{col} Range", float(data[col].min()), float(data[col].max()), (float(data[col].min()), float(data[col].max())))
                    min_max_dict[col] = (min_value, max_value)
            
            if st.button("Filter Data"):
                filtered_data = preprocess_data(data, exclude_keywords, include_keywords, exclude_urls, include_urls, min_max_dict)
                st.write("Filtered Data:")
                st.write(filtered_data)
                st.write(f'Row count after filtering: {len(filtered_data)}')
                st.session_state['filtered_data'] = filtered_data  
                st.session_state['filtered_keywords'] = filtered_data['Keyword']

                # Download link for filtered data
                filtered_csv = filtered_data.to_csv(index=False) 
                csv_bytes = base64.b64encode(filtered_csv.encode()).decode()
                st.download_button(
                    label="Download Filtered Data",
                    data=csv_bytes,
                    file_name="filtered_data.csv",
                    mime="text/csv"
                )

    if 'filtered_data' in st.session_state:
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
                    if 'filtered_keywords' in st.session_state:
                        keywords_to_query = st.session_state['filtered_keywords']
                        # Run DataForSEO API query for multiple keywords
                        result_df = query_dataforseo_serp(username, password, keywords_to_query)
                        st.session_state['result_df'] = result_df

                        if result_df is not None:
                            st.write("Data from DataForSEO API:")
                            st.dataframe(result_df)

                            #Adding a download button for the SERP similarity matrix
                            result_csv = result_df.to_csv(index=False)  
                            results_b64 = base64.b64encode(result_csv.encode()).decode()
                            st.download_button(
                                label="Download SERP Data",
                                data=results_b64,
                                file_name="Keyword-SERP-Data.csv",
                                mime="text/csv"
                            )

                        else:
                            st.error("Error querying DataForSEO API.")
                    else:
                        st.warning("No filtered keyword data available. Fetch keyword data from GSC first.")

    with st.expander("SERP Similarity"):
        if 'result_df' in st.session_state:
            selected_columns = st.session_state['result_df'][['Keyword', 'Keyword Intent', 'URL']]
            merged_df = pd.merge(st.session_state['filtered_data'], selected_columns, on='Keyword', how='inner')
            st.session_state['merged_df'] = merged_df

        if 'merged_df' in st.session_state:
            clusters = serps_similarity(merged_df)
            st.session_state['cluster_sim'] = clusters

        if 'cluster_sim' in st.session_state:
            if 'clicks' in st.session_state['filtered_data'].columns and 'impressions' in st.session_state['filtered_data'].columns:
                selected_columns1 = st.session_state['result_df'][['Keyword', 'Keyword Intent']]
                selected_columns2 = st.session_state['filtered_data'][['Keyword', 'clicks', 'impressions', 'ctr', 'position']]
                merged_df2 = pd.merge(selected_columns2, selected_columns1, on='Keyword', how='inner').drop_duplicates()
                agg_clusters = aggregate_clusters(clusters, merged_df2)
                st.write(agg_clusters)

            elif 'Search Volume' in st.session_state['filtered_data'].columns:
                selected_columns1 = st.session_state['result_df'][['Keyword', 'Keyword Intent']]
                selected_columns2 = st.session_state['filtered_data'][['Keyword', 'Search Volume']]
                merged_df2 = pd.merge(selected_columns2, selected_columns1, on='Keyword', how='inner').drop_duplicates()
                agg_clusters = aggregate_clusters(clusters, merged_df2)
                st.session_state['agg_clusters'] = agg_clusters
                st.write(agg_clusters)

        if 'agg_clusters' in st.session_state:

            #Adding a download button for the SERP similarity matrix
            cluster_csv = agg_clusters.to_csv(index=False)  
            cluster_b64 = base64.b64encode(cluster_csv.encode()).decode()
            st.download_button(
                label="Download Keyword Clusters",
                data=cluster_b64,
                file_name="keyword-clusters.csv",
                mime="text/csv"
            )

    with st.expander("Viz"):
        if 'agg_clusters' in st.session_state:
            create_bubble_chart(agg_clusters, x_limit=10, y_limit=12, font_size=8)

st.markdown('''
<style>
.stApp header {
    left: 100%;
}
</style>
''', unsafe_allow_html=True)

st.markdown('<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.3.1/dist/css/bootstrap.min.css" integrity="sha384-ggOyR0iXCbMQv3Xipma34MD+dH/1fQ784/j6cY/iJTQUOhcWr7x9JvoRxT2MZw1T" crossorigin="anonymous">', unsafe_allow_html=True)

st.markdown("""
<nav class="navbar fixed-top navbar-expand-lg" style="background-color: #c73c3c;">
    <a class="navbar-brand" href="https://tombras.com/"><img src="https://tombras.com/images/og.png" height=10% width=10% /></img></a>
</nav>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
