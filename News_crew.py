import os
import re
from datetime import datetime
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool

load_dotenv()
for key in ('OPENAI_API_KEY', 'SERPER_API_KEY'):
    if not os.getenv(key):
        raise RuntimeError(f'Missing {key} in .env')

today = datetime.now().strftime('%B %d, %Y')
llm = LLM(model=os.getenv('MODEL', 'gpt-4o-mini'),
          api_key=os.getenv('OPENAI_API_KEY'), temperature=0.1)

# The news search type is supported by current SerperDevTool versions.
# If your version rejects search_type, upgrade: pip install -U crewai-tools
search = SerperDevTool(search_type='news', n_results=10)

researcher = Agent(
    role='Technology News Researcher',
    goal='Find three distinct recent, source-backed technology news events.',
    backstory='You are a fact-checking technology reporter. You never invent sources.',
    tools=[search], llm=llm, verbose=True, max_iter=12,
    allow_delegation=False,
)
writer = Agent(
    role='Technology Newsletter Writer',
    goal='Write a factual newsletter of exactly five non-empty lines.',
    backstory='You are an editor who only uses the researcher findings.',
    tools=[], llm=llm, verbose=True, allow_delegation=False,
)

find = Task(
    description=(
        f'Today is {today}. You MUST call SerperDevTool to search for recent '
        'technology NEWS, using at least three different queries covering AI, '
        'cybersecurity, and cloud/software/major technology announcements. '
        'Choose three DIFFERENT specific news events from individual article URLs '
        '(not homepages, category pages, forecasts, or general trend articles). '
        'Prefer the last 48 hours; extend to the past 7 days if necessary. '
        'For each, give headline, publisher, article URL, reported publication '
        'date (or "date not shown"), and a factual one-sentence summary. '
        'Do not claim a publication date that is not shown in search results. '
        'If three credible stories cannot be found, return fewer and explain '
        'the limitation. Never invent articles, URLs, or dates.'
    ),
    expected_output=(
        'A numbered list of up to three distinct recent news events. Each entry '
        'includes a specific article URL, publisher, publication date or '
        '"date not shown", and a one-sentence summary.'
    ),
    agent=researcher,
)
write = Task(
    description=(
        'Use ONLY the preceding research task as evidence. Produce EXACTLY five '
        'non-empty lines, without markdown or blank lines. '
        'Line 1 must be "Tech in 5 Lines". Lines 2-4 must start "1. ", '
        '"2. ", "3. " and each summarize a DIFFERENT researched story. '
        'Line 5 starts "Why it matters: " and offers a factual takeaway. '
        'If the research has fewer than three actual stories, put '
        '"No verified story available" in the missing numbered slot. '
        'Never invent news or infer a slowdown from missing search results.'
    ),
    expected_output='Exactly five non-empty lines: heading, three numbered stories, takeaway.',
    agent=writer, context=[find],
)

crew = Crew(agents=[researcher, writer], tasks=[find, write],
            process=Process.sequential, verbose=True)


def validate(newsletter, research):
    lines = [line.strip() for line in newsletter.splitlines() if line.strip()]
    errors = []
    if len(lines) != 5:
        errors.append(f'Expected 5 non-empty lines, received {len(lines)}')
        return lines, errors
    if lines[0].rstrip('.') != 'Tech in 5 Lines':
        errors.append('Incorrect heading')
    for i in range(1, 4):
        if not re.match(rf'^{i}\.\s+\S', lines[i]):
            errors.append(f'Story {i} is blank or incorrectly numbered')
        if 'no verified story' in lines[i].lower() or 'no additional' in lines[i].lower():
            errors.append(f'Story {i} is missing')
    if not re.match(r'^Why it matters:\s+\S', lines[4]):
        errors.append('Missing takeaway')
    urls = set(re.findall(r'https?://[^\s)\]>]+', research))
    if len(urls) < 3:
        errors.append(f'Research contains only {len(urls)} distinct URL(s); need 3 article sources')
    return lines, errors


if __name__ == '__main__':
    print(f'\nResearching technology news as of {today}...\n')
    result = crew.kickoff()
    research = str(find.output)
    newsletter = str(result).strip()
    lines, errors = validate(newsletter, research)

    print('\n========== RESEARCHER OUTPUT ==========')
    print(research)
    print('\n========== FINAL NEWSLETTER ==========')
    print('\n'.join(lines))
    print('======================================')
    if errors:
        print('\nVALIDATION FAILED:')
        for error in errors:
            print('-', error)
        print('Review the researcher search results; do not submit unverified headlines.')
    else:
        print('\nFORMAT AND SOURCE-COUNT CHECKS PASSED.')
        print('Manually confirm each URL supports its headline and date before submitting.')
