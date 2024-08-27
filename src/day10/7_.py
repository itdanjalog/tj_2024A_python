import urllib.request
from bs4 import BeautifulSoup

base_url = "https://www.jobkorea.co.kr/Search/?stext="
search_term = "java"

for page in range( 1 , 200 ) :
    response = urllib.request.urlopen(f"{base_url}{page}")
    if response.getcode() != 200:
        print("Can't request website")
    else:
        results = []
        soup = BeautifulSoup(response.read(), "html.parser")
        jobs = soup.select('.content-recruit  .list-item ')  # 직업 정보(덩어리)가 있는 큰 틀(섹션) 찾기
        #print( jobs )
        for job_section in jobs:
            job_posts = job_section.select_one(".information-title")  # 조금 더 작은 범위로 들어가기
            # print(job_posts)
            print( job_posts.select('a')[0].text.strip() )