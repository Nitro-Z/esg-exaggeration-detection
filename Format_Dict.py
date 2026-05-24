words = """
dedicated to
positive impact
sustainability vision
superior
exceptional
exclusive
premium
extremely
ultimate
green innovation
greatly
super
perfect
best practices
vigorous
landmark
firmly believe
huge
remarkable
thriving
zero tolerance
highest standards
extraordinary
top-level
industry-leading
pioneering
attach great importance
surge
unconventional
record high
transformative
endless
clean technology
remain committed
infinite
we commit
exquisite
historic
fully committed
exciting
giant
we aspire
supreme
firmly committed
we endeavor
tremendous
top-tier
working towards
zero deforestation
world-leading
explosive
proud of
at the forefront
deeply committed
incredible
profound impact
significant progress
countless
we pledge
top-notch
wholeheartedly
striving for excellence
state-of-the-art
take the lead
actively pursuing
boundless
pursuing excellence
unforgettable
passionate about
comprehensive approach
glorious
ubiquitous
best-in-class
enormous
forward-thinking
important responsibility
eye-catching
golden age
sustainability journey
stirring
disruptive
exclusive to
spike
making progress
overwhelming
to the fullest
undoubtedly
to the extreme
significant contribution
integrated approach
everlasting
strongly believe
superb
splendid
sweep
magnificent
mysterious
spectacular
innovative approach
unstoppable
sustainability excellence
luxurious
captivating
relentless pursuit
ingenious
lasting impact
top ranking
meaningful contribution
deeply believe
steadfastly committed
dazzling
great achievements
fastest growing
making strides
first-rate
setting the standard
real impact
lead the market
epic
important mission
unique charm
robust framework
lead the trend
climate-friendly
worldwide attention
fearless
nature-friendly
full of vitality
leading sustainability
making a difference
commit to continuous
known to all
myriad
undisputed
green solution
ultimate experience
proud to announce
ahead of the curve
to the utmost
far more than
extremely difficult
unwavering dedication
transcendent
substantial improvement
tremendous impact
thrilling
irresistible
immeasurable
beyond normal
sustainable solution
driving change
real difference
premium grade
first-mover advantage
transformational impact
golden era
recognized leader
masterful
unimaginable
undisputed leader
playing a leading role
nurturing the future
sensational
tirelessly working
gaining momentum
red-hot
trailblazing
game-changing
making more contributions
indescribable
truly exceptional
incomparable
exceedingly
going above and beyond
rock-solid
raising the bar
highly creative
holistic strategy





""".strip().split("\n")

result = ", ".join(f'"{w.lower()}"' for w in words)
print(result)