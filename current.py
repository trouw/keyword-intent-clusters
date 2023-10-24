def query_dataforseo_serp(username, password, keywords, search_engine="google", search_type="organic", language_code="en", location_code=2840):
    # Define SERP features and their corresponding intent values
    feature_mapping = {
        "shopping": 0,
        "commercial_units": 1,
        "paid": 2,
        "popular_products": 2,
        "local_services": 2,
        "google_hotels": 2,
        "google_reviews": 2.5,
        "stocks_box": 2.5,
        "local_pack": 4,
        "hotels_pack": 4,
        "top_sights": 4,
        "google_flights": 4,
        "map": 5,
        "twitter": 5,
        "currency_box": 5,
        "explore_brands": 5,
        "events": 6,
        "mention_carousel": 6,
        "podcasts": 6,
        "jobs": 6.5,
        "app": 7,
        "multi_carousel": 7,
        "math_solver": 7,
        "visual_stories": 7,
        "images": 7.5,
        "related_searches": 7.5,
        "people_also_search": 7.5,
        "recipes": 7.5,
        "find_results_on": 7.5,
        "found_on_the_web": 7.5,
        "refine_products": 7.5,
        "carousel": 8,
        "top_stories": 8,
        "video": 8,
        "google_posts": 8.5,
        "knowledge_graph": 8.5,
        "people_also_ask": 9,
        "scholarly_articles": 9,
        "questions_and_answers": 9,
        "short_videos": 9,
        "answer_box": 10,
        "featured_snippet": 10
    }

    # Create a RestClient instance
    client = RestClient(username, password)
    all_data = []

    total_keywords = len(keywords)
    progress_bar = st.progress(0)  # Initialize progress bar

    # Prepare the task parameters for multiple keywords
    task_params = [
        {
            "language_code": language_code,
            "location_code": location_code,
            "keyword": keyword,
            "calculate_rectangles": True
        }
        for keyword in keywords
    ]

    endpoint = f"/v3/serp/{search_engine}/{search_type}/live/advanced"

    # Send a single API request for all keywords
    response = client.post(endpoint, task_params)

    if response["status_code"] == 20000:
        # Iterate through the response to extract data for each keyword
        for index, keyword in enumerate(keywords):
            keyword_results = response['tasks'][index]['result'][0]

            keyword_intent = []
            # Update progress bar
            progress_bar.progress((index + 1) / total_keywords)

            # Extract SERP features
            for i in keyword_results['item_types']:
                if i in feature_mapping:
                    keyword_intent.append(feature_mapping[i])

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

            # Create a list to store the extracted data for this keyword
            data_list = []

            # Iterate through the organic results and extract relevant information
            for result in organic_results:
                url = result.get('url')
                position = result.get('rank_absolute')
                title = result.get('title')
                description = result.get('description')

                # Append the extracted data to the list
                data_list.append([keyword, url, position, title, description, intent_avg])

            # Add the data for this keyword to the list of all data
            all_data.extend(data_list)

        else:
            print(f"Error for keyword '{keyword}': Code {response['status_code']} Message: {response['status_message']}")

    # Create a DataFrame from the combined data for all keywords
    df = pd.DataFrame(all_data, columns=["Keyword", "URL", "Position", "Title", "Description", "Keyword Intent"])

    return df
