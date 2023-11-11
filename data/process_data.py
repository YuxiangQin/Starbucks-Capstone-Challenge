import sys
import pandas as pd
import heapq
import json
from sqlalchemy import create_engine

def load_data(portfolio_filepath, profile_filepath, transcript_filepath):
    '''Load data from input filepaths and return three pandas dataframes'''
    portfolio = pd.read_json(portfolio_filepath, orient='records', lines=True)
    profile = pd.read_json(profile_filepath, orient='records', lines=True)
    transcript = pd.read_json(transcript_filepath, orient='records', lines=True)
    return portfolio, profile, transcript

def clean_profile(df):
    '''Clean profile data'''
    df = df[df['age'] != 118].copy() # drop noise rows
    df['became_member_on_dt'] = pd.to_datetime(df['became_member_on'], format="%Y%m%d")
    max_became_member_date = df['became_member_on_dt'].max()
    df['tenure_day'] = (max_became_member_date - df['became_member_on_dt']).dt.days
    df['tenure_month'] = df['tenure_day'] // 30
    df['tenure_year'] = df['tenure_day'] // 365
    df.drop(columns=['became_member_on', 'became_member_on_dt'], inplace=True)
    df.rename(columns={'id': 'customer_id'}, inplace=True)
    return df

def clean_portfolio(df):
    '''Clean portfolio data'''
    # transform channels to separate columns
    for index, row in df.iterrows():
        for channel in row['channels']:
            if channel not in df.columns: # add new channel column
                df[channel] = 0
            df.at[index, channel] = 1

    df.drop(columns=['channels'], inplace=True)
    df.rename(columns={'id': 'offer_id'}, inplace=True)
    return df

def clean_transcript(df):
    '''Clean transcript data'''
    def extract_first_value(dictionary):
        '''Extract first value of input dictionary'''
        if isinstance(dictionary, dict) and dictionary:  # Check if non-empty dictionary
            return next(iter(dictionary.values()))
        else:
            return None

    df['value_key'] = df['value'].apply(lambda x: list(x.keys())[0])
    df['value_key'] = df['value_key'].str.replace(' ','_') # "offer_id" and "offer id"

    df['offer_id'] = df[df['value_key'] == 'offer_id']['value'].apply(extract_first_value)
    df['transaction_amount'] = df[df['value_key'] == 'amount']['value'].apply(extract_first_value)
    df.rename(columns={'person': 'customer_id'}, inplace=True)
    return df

def clean_data(portfolio, profile, transcript):
    return clean_portfolio(portfolio), clean_profile(profile), clean_transcript(transcript)

def add_offer_received_row(customer_offer_data, customer, offer_id, offer_type, time):
    '''Add a new row to customer_offer_data'''
    new_record = pd.DataFrame([{"customer": customer,
                                "offer_id": offer_id,
                                "offer_type": offer_type,
                                "offer_receive_time": time,
                                "offer_view_time": None,
                                "offer_complete_time": None}])
    return pd.concat([customer_offer_data, new_record], axis=0, ignore_index=True)

def update_offer_activity_time(customer_offer_data, customer, offer_id, time, activity_nm):
    '''
    Update offer activity time in customer_offer_data
    For @customer, @offer_id, update most recent blank @activity_nm to @time.
    '''
    update_row = customer_offer_data[(customer_offer_data["customer"] == customer) &
                                       (customer_offer_data["offer_id"] == offer_id) &
                                       (customer_offer_data[activity_nm].isnull())]
    if not update_row.empty:
        update_row_index = update_row.index[-1]  # select most recent row to update
        customer_offer_data.at[update_row_index, activity_nm] = time

def add_to_offer_pq(offer_pq, time, offer_id, offer_type, view_time, duration):
    '''Add a new record to offer_pq'''
    heapq.heappush(offer_pq, [time + duration * 24, offer_id, offer_type, view_time])

def remove_from_offer_pq(offer_pq, offer_id):
    '''Remove @offer_id from offer_pq'''
    return [offer for offer in offer_pq if offer[1] != offer_id]

def get_impact_offer_info(offer_pq: heapq, time: int) -> tuple:
    '''
    Get offer impact information for a transaction happened at input time.

    Returned impact_offer_id is always one item, if multiple offers impact a transaction:
    - Choose non-informational offer over informational offer
    - Multiple non-informational offers, choose the one with earliest impact end time
    - Multiple informational offers, choose the one with latest impact end time

    Args:
        offer_pq (heapq): Priority Queue of offers
        time (int): Transaction time

    Returns:
        Tuple containing:
        - impact_offer_id (str): offer_id impacted the transaction
        - impact_informational_offers (list): A list of all informational offer_ids impact this transation
        - has_non_informational_offer (bool): If there's non-informational offer impacted this transaction
    '''
    # pop out expired offers
    while offer_pq and offer_pq[0][0] < time:
        heapq.heappop(offer_pq)

    impact_offer_id = None
    impact_informational_offers = []
    has_non_informational_offer = False

    for offer in offer_pq:
        impact_offer_id = offer[1]
        if offer[2] == 'informational':
            impact_informational_offers.append(offer)
        else:
            has_non_informational_offer = True
            break
    return impact_offer_id, impact_informational_offers, has_non_informational_offer

def add_informational_respond(customer_offer_data, customer, impact_informational_offers):
    '''Add respond flag to customer_offer_data for informational offers'''
    if impact_informational_offers:
        for info_offer in impact_informational_offers:
            impact_row = customer_offer_data[(customer_offer_data['customer'] == customer) &
                                   (customer_offer_data['offer_id'] == info_offer[1]) &
                                   (customer_offer_data['offer_view_time'] == info_offer[3])]
            customer_offer_data.at[impact_row.index[0], "offer_respond"] = 1

def fill_offer_respond(row):
    '''Fill offer_respond column base on conditions'''
    if row['offer_respond'] == 1 or (
        row["offer_type"] != "informational" and
        row["offer_complete_time"] is not None and
        row["offer_view_time"] is not None and
        row["offer_complete_time"] >= row["offer_view_time"]
    ):
        return 1
    else:
        return 0

def get_customer_respond(df):
    '''
    Returns customer-offer-view_time level data with offer_respond flag.
    Meanwhile add a column of impact_offer_id to input df.
    '''
    customer_offer_data = pd.DataFrame(columns=["customer", "offer_id", "offer_type", "offer_receive_time",
                                                "offer_view_time", "offer_complete_time", "offer_respond"])

    for customer in df['customer_id'].unique():
        customer_data = df[df['customer_id'] == customer]
        offer_pq = []

        for index, row in customer_data.iterrows():
            event = row["event"]
            offer_id = row["offer_id"]
            offer_type = row["offer_type"]
            time = row["time"]
            duration = row["duration"]

            if event == "offer received":
                customer_offer_data = add_offer_received_row(customer_offer_data, customer, offer_id, offer_type, time)

            elif event == "offer viewed":
                update_offer_activity_time(customer_offer_data, customer, offer_id, time, "offer_view_time")
                add_to_offer_pq(offer_pq, time, offer_id, offer_type, time, duration)

            elif event == "offer completed":
                update_offer_activity_time(customer_offer_data, customer, offer_id, time, "offer_complete_time")
                offer_pq = remove_from_offer_pq(offer_pq, offer_id)

            elif event == "transaction":
                impact_offer_id, impact_informational_offers, has_non_informational_offer = get_impact_offer_info(offer_pq, time)
                # record respond to informational offers
                if not has_non_informational_offer:
                    add_informational_respond(customer_offer_data, customer, impact_informational_offers)
                # record impact offer_id
                if impact_offer_id:
                    df.at[index, 'impact_offer_id'] = impact_offer_id

    # Fill offer_respond column
    customer_offer_data['offer_respond'] = customer_offer_data.apply(fill_offer_respond, axis=1)
    customer_offer_data.rename(columns={'customer': 'customer_id'}, inplace=True)

    return customer_offer_data

def get_customer_transaction_info(df):
    '''Get customer-with_offer level aggregated transaction information.'''
    no_offer = df[(df['value_key'] == 'amount') & (df['impact_offer_id'].isnull())]
    no_offer = no_offer.groupby('customer_id').agg({'transaction_amount':'sum',
                                                'event': 'count'}).reset_index()
    no_offer['with_offer'] = 0
    with_offer = df[(df['value_key'] == 'amount') & (df['impact_offer_id'].notnull())]
    with_offer = with_offer.groupby('customer_id').agg({'transaction_amount':'sum',
                                                'event': 'count'}).reset_index()
    with_offer['with_offer'] = 1
    res = pd.concat([no_offer, with_offer], axis=0)
    res.rename(columns={'event': 'transaction_count'}, inplace=True)

    return res

def save_data(df, database_filename):
    '''save @df as @database_filename'''
    engine = create_engine(f"sqlite:///{database_filename}")
    df.to_sql('data', engine, index=False, if_exists='replace')

def main():
    if len(sys.argv) == 6:
        portfolio_filepath, profile_filepath, transcript_filepath,\
        database1_filepath, database2_filepath = sys.argv[1:]

        print(f"Loading data...\n"
              f"Portfolio: {portfolio_filepath}\n"
              f"Profile: {profile_filepath}\n"
              f"Transcript: {transcript_filepath}")
        portfolio, profile, transcript = load_data(portfolio_filepath, profile_filepath, transcript_filepath)

        print("Cleaning data...")
        portfolio, profile, transcript = clean_data(portfolio, profile, transcript)
        merged = transcript.merge(profile, on='customer_id', how='inner') \
                       .merge(portfolio, on='offer_id', how='left')

        print("Generating customer respond...")
        customer_offer_data = get_customer_respond(merged)
        # add customer and offer features
        customer_offer_data.drop(columns=['offer_type'], inplace=True)
        customer_offer_data = customer_offer_data.merge(profile, on='customer_id', how='inner') \
                       .merge(portfolio, on='offer_id', how='left')

        print('Saving data...\n    DATABASE: {}'.format(database1_filepath))
        save_data(customer_offer_data, database1_filepath)

        print('Saving data...\n    DATABASE: {}'.format(database2_filepath))
        cust_trans_agg = get_customer_transaction_info(merged)
        save_data(cust_trans_agg, database2_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the portfolio, profile and transcript '\
              'datasets as the first, second and third argument respectively, as '\
              'well as the filepaths of the database to save the cleaned data '\
              'to as the forth and fifth argument. \n\nExample: python process_data.py '\
              'portfolio.json profile.json transcript.json '\
              'customer_offer_data.db customer_trans_agg.db')

if __name__ == '__main__':
    main()
