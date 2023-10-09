import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

def get_demonym(city):
    # Construct the Wikipedia link for the given city
    base_url = "https://en.wikipedia.org/wiki/"
    city_url = base_url + city

    # Send a GET request to fetch the HTML content of the city's Wikipedia page
    response = requests.get(city_url)
    if response.status_code != 200:
        print(f"Failed to fetch {city} Wikipedia page.")
        return "Failed_to_Fetch"

    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.text, "html.parser")

    # Find the demonym information using the specified <th> and <td> tags
    demonym_element = soup.find("a", title="Demonym")
    if demonym_element:
        demonym_value = demonym_element.find_next("td")
        if demonym_value:
            return demonym_value.text.strip()
    
    # print(f"No demonym information found for {city}.")
    return None

# Define a function to process a single city entry
def process_city_entry(city_entry):
    city_name, city_ascii, country = city_entry
    city_name = city_name.lower()
    city_name = city_name.strip()
    city_name = city_name.replace(' ', '_')
    demonym = get_demonym(city_name)
    
    if demonym !="Failed_to_Fetch":
        return city_ascii, demonym, country
    else:
        demonym = get_demonym(city_ascii)
        if demonym == "Failed_to_Fetch":
            return f"{city_ascii}___{country}",None,None
        elif demonym:
            return city_ascii, demonym, country
        else:
            return city_ascii, None, country

def main():

    df = pd.read_csv("worldcities.csv")

    # Create a list of tuples containing city names and countries
    city_info_list = [(city, city_asci, country) for city,city_asci, country in zip(df['city'], df['city_ascii'],df['country'])]
    # city_info_list = city_info_list[:1000]
    # Define the number of CPU cores you want to utilize
    num_cores = multiprocessing.cpu_count()

    # Split the city_info_list into chunks for parallel processing
    chunk_size = len(city_info_list) // num_cores
    chunks = [city_info_list[i:i + chunk_size] for i in range(0, len(city_info_list), chunk_size)]

    # Initialize lists to collect results
    demonym_cities = []
    cities_name = []
    country = []
    not_found_cities = []
    not_found_country = []
    failed_to_fetch = []
    # Create a multiprocessing Pool
    with multiprocessing.Pool(processes=num_cores) as pool:
        results = list(tqdm(pool.imap(process_city_entry, city_info_list), total=len(city_info_list)))

    # Separate the results into found and not found
    for result in results:
        city_name, demonym, city_country = result
        if demonym:
            demonym_cities.append(demonym)
            cities_name.append(city_name)
            country.append(city_country)
        elif city_country==None:
            failed_to_fetch.append(city_name)
        else:
            not_found_cities.append(city_name)
            not_found_country.append(city_country)

    # Create DataFrames for the found and not found cities
    output_df = pd.DataFrame(data={"City": cities_name, "Inhabitant_term": demonym_cities, "Country": country})
    output_df.to_csv("dataset_cities_inhabitant_term_asci.csv", index=False)

    not_found_city_df = pd.DataFrame({"City": not_found_cities, "Country": not_found_country})
    not_found_city_df.to_csv("Not_found_city_asci.csv", index=False)
    
    failed_to_fetch_df = pd.DataFrame({"City": failed_to_fetch})
    failed_to_fetch_df.to_csv("Failed_to_Fetch_city_asci.csv", index=False)


if __name__=="__main__":
    main()
    # Example usage
    
