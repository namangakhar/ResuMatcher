import os
from dotenv import load_dotenv
from google.adk.agents import Agent, SequentialAgent
from google.adk.tools.tool_context import ToolContext

load_dotenv()

model_name = os.getenv("MODEL", "gemini-2.5-flash")


# --- Tool: Read resume from file ---
def read_resume_file(tool_context: ToolContext) -> dict:
    """Reads the candidate's resume from resume.txt file and stores it in state.

    Returns:
        The resume text or an error message.
    """
    try:
        with open("resume.txt", "r") as f:
            resume_text = f.read()
        tool_context.state["RESUME"] = resume_text
        return {"status": "success", "message": "Resume loaded from resume.txt", "preview": resume_text[:200] + "..."}
    except FileNotFoundError:
        return {"status": "error", "message": "resume.txt not found. Please place resume.txt in the project directory."}


# --- Tool: Read JD from file ---
def read_jd_file(tool_context: ToolContext) -> dict:
    """Reads the job description from jd.txt file and stores it in state.

    Returns:
        The JD text or an error message.
    """
    try:
        with open("jd.txt", "r") as f:
            jd_text = f.read()
        tool_context.state["JD"] = jd_text
        return {"status": "success", "message": "JD loaded from jd.txt", "preview": jd_text[:200] + "..."}
    except FileNotFoundError:
        return {"status": "error", "message": "jd.txt not found. Please place jd.txt in the project directory."}


# --- Tool: Save resume text to state (for paste input) ---
def save_resume(tool_context: ToolContext, resume_text: str) -> dict:
    """Saves the candidate's resume text to state.

    Args:
        tool_context: The tool context for state management.
        resume_text: The candidate's full resume or LinkedIn profile text.

    Returns:
        A confirmation that the resume has been stored.
    """
    tool_context.state["RESUME"] = resume_text
    return {"status": "success", "message": "Resume saved."}


# --- Tool: Save JD text to state (for paste input) ---
def save_jd(tool_context: ToolContext, job_description: str) -> dict:
    """Saves the job description to state.

    Args:
        tool_context: The tool context for state management.
        job_description: The complete job description to match against.

    Returns:
        A confirmation that the JD has been stored.
    """
    tool_context.state["JD"] = job_description
    return {"status": "success", "message": "JD saved."}


# --- Step 1: Analyzer ---
analyzer_agent = Agent(
    name="analyzer",
    model=model_name,
    description="Analyzes resume against JD and produces match report.",
    instruction="""You are a resume-JD matching expert.

You MUST analyze the following RESUME against the following JD carefully.
Read both thoroughly before scoring.

RESUME:
{RESUME}

JD:
{JD}

Now produce your analysis in this EXACT format:

========================================
           MATCH ANALYSIS
========================================

**Overall Match Score:** [X/100]
**Verdict:** [Strong Match / Moderate Match / Weak Match - brief explanation]

**Strengths (Skills & Experience that Align):**
- [Specific strength referencing actual resume content mapped to JD requirement]
- [Continue for all matches]

**Gaps (What the JD Requires but Resume Lacks):**
- [Specific gap — with concrete suggestion to address it]
- [Continue for all gaps]

**ATS Keyword Suggestions:**
- [Keywords from the JD that should appear in the profile]

========================================

Rules:
- The score MUST reflect actual overlap. If resume has 60% of required skills, score around 60.
- Reference SPECIFIC skills from the resume text above. Do not say "no skills found".
- Be honest but fair.
""",
    output_key="match_analysis",
)


# --- Step 2: Cover Letter Writer ---
cover_letter_agent = Agent(
    name="cover_letter_writer",
    model=model_name,
    description="Generates a tailored cover letter.",
    instruction="""Write a professional cover letter for this candidate applying to this role.

RESUME:
{RESUME}

JD:
{JD}

MATCH ANALYSIS:
{match_analysis}

Format your output EXACTLY like this:

****************************************
         TAILORED COVER LETTER
****************************************

Dear Hiring Manager,

[Paragraph 1: Enthusiasm for the specific role + strongest qualification from resume]

[Paragraph 2: 2-3 specific achievements from resume mapped to JD requirements with metrics]

[Paragraph 3: Address 1-2 gaps positively as growth opportunities + cultural fit]

[Paragraph 4: Confident close + call to action]

Sincerely,
[Candidate Name from resume]

****************************************
""",
    output_key="cover_letter",
)


# --- Step 3: Resume Optimizer ---
resume_optimizer_agent = Agent(
    name="resume_optimizer",
    model=model_name,
    description="Rewrites and optimizes the resume for the target JD.",
    instruction="""Rewrite this candidate's resume tailored for the target JD.

RESUME:
{RESUME}

JD:
{JD}

MATCH ANALYSIS:
{match_analysis}

Format your output EXACTLY like this:

########################################
          OPTIMIZED RESUME
########################################

[CANDIDATE NAME]
[Email | Phone | LinkedIn | Location from original resume]

--- PROFESSIONAL SUMMARY ---
[3-4 lines. Lead with experience + core expertise. Include top JD keywords naturally.]

--- TECHNICAL SKILLS ---
[Reorder to prioritize JD-relevant skills FIRST.
Mark any suggested additions with [SUGGESTED].
Group by: Languages | Frameworks | Cloud & DevOps | Databases | Tools]

--- PROFESSIONAL EXPERIENCE ---

[Job Title] | [Company] | [Dates]
* [Rewritten bullet emphasizing JD-relevant achievement with metrics]
* [Continue for each bullet]

[Repeat for each role]

--- EDUCATION ---
[Degree, University, Year]

--- SUGGESTED CERTIFICATIONS ---
[2-3 certifications the candidate should pursue based on gap analysis]

########################################

Rules:
- NEVER fabricate experience or skills not in the original resume
- Mark suggested additions with [SUGGESTED]
- Weave ATS keywords from the JD naturally throughout
- Make it immediately copy-pasteable
""",
    output_key="optimized_resume",
)


# --- Step 4: Formatter ---
formatter_agent = Agent(
    name="formatter",
    model=model_name,
    description="Assembles all outputs into one clean response.",
    instruction="""Present ALL the work from previous agents as one clean response.

Start with this line:
"Here is your complete ResuMatch analysis! You will find your match report, a tailored cover letter, and an optimized resume below — all ready to use."

Then paste each section in this order with NO changes:
1. The Match Analysis
2. The Cover Letter
3. The Optimized Resume

End with:
"Need changes? Just ask — I can adjust the tone, focus on different skills, or rewrite any section!"

IMPORTANT: Copy each section EXACTLY as provided. Do NOT summarize, truncate, or re-analyze.

MATCH ANALYSIS:
{match_analysis}

COVER LETTER:
{cover_letter}

OPTIMIZED RESUME:
{optimized_resume}
""",
)


# --- Pipeline ---
analysis_pipeline = SequentialAgent(
    name="analysis_pipeline",
    description="Runs full analysis: match scoring, cover letter, resume optimization, formatting.",
    sub_agents=[
        analyzer_agent,
        cover_letter_agent,
        resume_optimizer_agent,
        formatter_agent,
    ],
)


# --- Root Agent: collects resume + JD from files or paste ---
root_agent = Agent(
    name="resumatch_agent",
    model=model_name,
    description="ResuMatch — collects resume and JD, then runs analysis pipeline.",
    instruction="""You are ResuMatch, a friendly AI career advisor.

You collect the resume and job description, then hand off to the analysis pipeline.

YOU HAVE THESE TOOLS:
- read_resume_file: Reads resume from resume.txt file
- read_jd_file: Reads JD from jd.txt file
- save_resume: Saves pasted resume text to state
- save_jd: Saves pasted JD text to state

FOLLOW THESE STEPS:

STEP 1: Greet the user. Ask how they want to provide their resume:
  a) Type "analyze" or "go" to load from resume.txt and jd.txt files automatically
  b) Paste resume text directly

STEP 2a (files): If user says "analyze", "go", "run", "start", or "load files":
  - Call read_resume_file to load resume.txt
  - Call read_jd_file to load jd.txt
  - If both succeed, IMMEDIATELY transfer to analysis_pipeline

STEP 2b (paste): If user pastes resume text:
  - Call save_resume with their text
  - Ask for the JD
  - When they provide JD, call save_jd
  - Then IMMEDIATELY transfer to analysis_pipeline

RULES:
- After both resume and JD are in state, IMMEDIATELY transfer to analysis_pipeline.
- Do NOT analyze anything yourself.
- Do NOT ask for inputs twice.
""",
    tools=[read_resume_file, read_jd_file, save_resume, save_jd],
    sub_agents=[analysis_pipeline],
)
