import pandas as pd
import requests

def get_local_code( fname = "PTS/accessheader.txt" ) :
    accessheader = ""
    with open(fname,'r') as input :
        for line in input :
            accessheader = line.replace("\n","")
            break
    return accessheader

accessheader = get_local_code()

def get_api_token( url = "https://viewbackend.pts.se/api/v1/Token/GetToken" ,
                   access_header_value = accessheader ) :
    #
    headers = {
        "accessheader": access_header_value
    }

    # Make the GET request
    response = requests.get(url, headers=headers)

    # Raise an exception if something goes wrong
    response.raise_for_status()

    # Print the token (assuming it's returned as JSON)
    API_token = response.text

    return API_token

repo = {
    "Frekvensplan":"https://viewbackend.pts.se/api/FrqPlan",
    "Undantag":"https://viewbackend.pts.se/api/LicenseExcemption",
    "Undantag spec":"https://viewbackend.pts.se/api/LicenseExcemption/", #NUMMER
    "Exempel":"https://viewbackend.pts.se/api/LicenseExcemption/1435"
}

def retrieve_information(url,token,acessheader,bVerbose=True):
    api_url = url 
    headers = {
        "access-token": token,
        "accessheader": accessheader
    }
    data_response = requests.get(api_url, headers=headers)
    if bVerbose:
        show_response ( data_response )
    return data_response


def show_response(data_response):
    print( data_response.json() )

def response_dataframe(data_response):
    return ( pd.DataFrame( [ list(m.values()) for m in data_response.json()],
					columns = list(data_response.json()[0].keys()) ) )

def levenshtein_single(a: str, b: str) -> int:
    if len(a) > len(b):
        a, b = b, a
    
    prev = list(range(len(a) + 1))
    for i, bc in enumerate(b, 1):
        curr = [i]
        for j, ac in enumerate(a, 1):
            insert = curr[j - 1] + 1
            delete = prev[j] + 1
            replace = prev[j - 1] + (ac != bc)
            
            if insert < delete:
                best = insert
            else:
                best = delete
            if replace < best:
                best = replace
            curr.append(best)
        prev = curr
    return prev[-1]


if __name__ == '__main__':

    bReset = True
    if bReset :
        ATOKEN = get_api_token()
        o_f = open("PTS/TOKEN.txt",'w')
        print(ATOKEN,file=o_f)
        o_f.close()

    ATOKEN = get_local_code( fname = "PTS/TOKEN.txt" )
    token  = ATOKEN
    #
    # 2. Använd token för att göra anrop
    freqp = retrieve_information( url = repo["Frekvensplan"] ,
				  token=token , acessheader=accessheader ,
				  bVerbose = False )
    show_response ( freqp ) 
    freq_df = response_dataframe( freqp )
    freq_df .to_csv('frekvensplan_via_api.csv',sep=';')

    undantag = retrieve_information(	url = repo["Undantag"] ,
				token=token , acessheader=accessheader ,
				bVerbose = False )
    #
    # WRITE JSON FILES
    #
    ofile = open('freqp.json','w')
    print(freqp.json(),file=ofile)
    ofile.close()
    #
    undantag_details = []
    for item in undantag.json() :
        print(item)
        utf_detail = retrieve_information(	url = repo["Undantag spec"] + str(item['id']) ,
				token=token , acessheader=accessheader,
				bVerbose = False )
        add_item=utf_detail.json()
        print(add_item)
        for key in add_item.keys():
            nkey = key
            if key in item :
                nkey = key+'.a'
            item[nkey]=add_item[key]
        undantag_details.append(item)
    ofile = open('undantag.json','w')
    print(undantag_details,file=ofile)
    ofile.close()    
