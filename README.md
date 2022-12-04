# Hallmark Philosophers

- [Data](#abstract)
- [Methods](#examples)
  * [Vertical Learning](#vertical-learning)
  * [Horizontal Learning](#horizontal-learning)
  * [Data Combination](#data-combination)
- [Sources](#sources)

![NietMark](/images/nietmark.png)

One thing we are for sure tired of is seeing the same Hallmark movie over and over again. Whether it’s in New York City or rural Idaho, we decided to make a new script that could change things up from the usual story and scriptline. This project goal was to create an entirely new movie, maybe even something to put Hallmark to shame, by combining the original structure of the movie script but adapting language and concepts from philosophers such as Nietzsche. Each approach uses an LSTM RNN to process such a large amount of data and extract important speech semantics and vernacular.

# Data

There are two main data sets used in this experiment: a movie text style script from Rick Garman, a writer and frequent Hallmark collaborator and raw text files from philosophers Neitzsche and Plato.

# Vertical Learning

![Vertical Image](/images/vertical.png)

In this “vertical” learning process, we used one model and train it with the hallmark movie text first, freeze its layers so it’s not trainable anymore, remove the last layer, add a new LSTM layer and the last layer, and make it train by Nietzsche’s text.

## First part
After 20 epochs of learning only the hallmark movie script, the generated script looked like this. You can see it has the movie script formatting such as putting the name and the character’s line one by one, and it has a little broken English vocabulary:

```
Establishing shot of Leigh & Erin’s apart
know.
LEIGH
I was you want to walk and herelf at the carry.
BRAD
Oh at his here his a compacter.
LEIGH
Yeah, I do?
ALLISON
(noges)
Then shakes at the pages and out on the table.
She faied you and seep in the ready. *
BRAD
So wast a petty me aly and then should prote thing that
21/21 [==============================] - 1430s 69s/step - loss: 1.3336 - categorical_crossentropy: 1.3336 - accuracy: 0. 5977
```

## Second part
Even though the model has learned from the hallmark movie, the first epoch looked like a gibberish, since it has a new text data imported. This is the first epoch of the second part of the vertical learning process:

```
reat that he actually does
sacrifice somet  ha   i  t s  tan  ht t c ie s  e  e  .e w n     reRe ao  t wh   erhst  aa ha Rd tse  e a o
 e es n i n  y. e n a i oio t     oaiwe ihe o wt  hks er   e    pt o I   o  se htehg . ae at
ci

t i e t n  hni .      s teag t e a rl i.e
di t e  if  o    n   
 ae  e .  ati e  a, t r   we  wn eo  w  f o atar r  reewaasintheu lycal  ar is e  hahi,
 eie  e h i ere
nnoeiey in i   elae
s ..o  h h d oa tyo cetha it 
ea weani  a wt  t
 a)seeyr    n t tenerhe wae f ?o 
 e a ea at at yaee ae to y   ’ee n 
mee  a bnt 
 ahe ey   rashad ega ee wheral in r r  i hae ta ni ocai w
I a  e     ree e e ce o 
 laor   eh   it reeoh e aat o hh  s y ie ae h H Tent  nreamhy  m  
u    t a a  ih g 
   et   le il. ehe.
a   e  i at  ou fei

o e  i a       t t   nhin     o Hn   tel hi te 
  a  en  n . ta  deiliee ’ a  n a ean  o e  et
 e ae hen   omeni on
raa  e niN   ht Ih en dt es 
n  na h t  
ei   r e w laewe 
h aea 
h .at 
  i  h  e o    y i  al ahen i  
Oeae 
a hre uta 
efra .
 ee atei   o   h  n ’t e n  n dsve  o   ia.noan   ta
 A ine i a r a   in o iaecit a   re th t
 e
r n
d e i e tt
o n e  e tce  ri   ityteie styow co  eee it  ii aale aat  Reer u.e e
```
And below is the last two epochs of this learning process. In the 19th epoch, you can see it clearly has the movie script formatting such as character name and line put one another, and scene discription at one point, saying “INT. RESTAURANT - NIGHT.” Also, in the 20th epoch, you can see phrases such as “Christianity” that are assumingly taken from nietzsche’s text.
```
21/21 [==============================] - 179s 9s/step - loss: 3.7099 - categorical_crossentropy: 3.7099 - accuracy: 0.1329


21/21 [==============================] - 190s 9s/step - loss: 1.3265 - categorical_crossentropy: 1.3265 - accuracy: 0.6000
Epoch 19/20
21/21 [==============================] - ETA: 0s - loss: 1.3005 - categorical_crossentropy: 1.3005 - accuracy: 0.6072
----- Generating text after Epoch: 18
----- diversity: 0.5
----- Generating with seed: "led the furthest with the nearest, fire "
led the furthest with the nearest, fire it to do start down a
dide. No.
ALLISON
And you talks it to good fire to the computer going to frea complet down at the computer *
Leigh is start of then INT. RESTAURANT - NIGHT
Erin is to see the stant to smile in the door and then just is puts dates up to do stop dates and em to thing that the computer *
well then she stance and he don’t got in to the door on the waiter, so.
ALLISON
I can’t eat of the door and Conner is the could *
and then it to the comple
first desk on the computer it to
the office it the start to me now reading with we wat he frong the door show the door *
like the date of a give. *
ALLISON
He think the resore what I
was the door date it to start of care for a moment, then start to say the rood to leilt.
ALLISON
And we rues and then it was just a date. Do you
me do.
ALLISON
And we stop in the know. I don’t you like where the table to the some his do she stop on the table to start to mealy.
ALLISON
And we starts a did there to inte from to the rest in the date with a something what you figure good and then the poop...
ALLISON
I dat? How you’re going to say thing to paper when I was the bat *
21/21 [==============================] - 197s 10s/step - loss: 1.3005 - categorical_crossentropy: 1.3005 - accuracy: 0.6072
Epoch 20/20
21/21 [==============================] - ETA: 0s - loss: 1.2748 - categorical_crossentropy: 1.2748 - accuracy: 0.6135
----- Generating text after Epoch: 19
----- diversity: 0.5
----- Generating with seed: "course, wherever Christianity prospers
a"
course, wherever Christianity prospers
and then it like that the was one of the could her to make it the could It’s the could on the car and the same toward the was a chote the concrett that
it is the kit it wo could think the book down the waiter the betted toward the car and to see the could *
explaition his thing and then the coult like this hat me a smert her to mean in the could
can *
the other and then the car and the could he have that
it’s eyters to sit of the car were
with the conches to sit it to the was a moment the could her acroompy. I’m sorry for the could *
the carmantice and the story conther the office in the chuight back and then the kitchen the has in the know whate you’re going to the couch other the table and the other her
the one of the could *
trying to see this we car a resk on fire to meen the could *
with to see the couch one of the good down to see this the bother bad date with the could to see this into the could the *
the cauchat she was the far the one of the car and take it care the door the car reading at the table this is the know we it to the could ton the too at to wore thing to be a
21/21 [==============================] - 258s 13s/step - loss: 1.2748 - categorical_crossentropy: 1.2748 - accuracy: 0.6135
```

# Horizontal Learning

![Horizontal_Image](/images/horizontal.png)

The second approach to learning was to independently train two models on our source texts, then use their outputs as seeds for one another in a cycle to generate text. This will be called “horizontal” learning as we take two models in parallel to work together, as opposed to adding pieces to one.

```
----- Generating with seed: "m the subtle and nothing of art again on"
m the subtle and nothing of art again on a done for a moment bad
dater going to the reatiply a computhing.
110 110 EXT. LEIGH AND ERIN’S APARTHTMETEH PCARTMENT - DAY
Leing smiles and then didn’t know dessers, this to the date *
like listle and funny her. She shot on she is not sitting read, I’ll sone of the computer not head happenty discles this and the cand. I’m not know what is reaches and stands a on of the table inet of the Post wh

----- Generating with seed: " a done for a moment bad
dater going to "
 a done for a moment bad
dater going to the races of utility, ever in the first virtues like one are was and there is only the _strong from the soul" at the explanation
spake to the consequence of reality. The world of the
period of the nob

----- Generating with seed: "the races of utility, ever in the first "
the races of utility, ever in the first going to do you have by befter him a finl way. What that. *
CONNER (V.O.)
Not this could look lookhant for a bad date. It’s that dinnecred of the Gook as I’m not to the computing an hore to wait both me anyon actuctirs from what you go nots to the table. and their are rell is starts to don’t going to do it you lifon.
LEIGH *
No...
Here ideal the table... Somors over the crecticles.
LEIGH
How did I

----- Generating with seed: "going to do you have by befter him a fin"
going to do you have by befter him a fine standing that the individual, and the songs and for the common for that this out of means of a large reading" of the sight of distances. As
two higher and yet carede, and soul is here that is the pe

```



# Data Combination

![Combined Data](/images/combined.png)

### Combined text file
With this method, both data files were spliced together so each line comes from an alternating source. To keep the data equal, the combination was stopped after the last line in the screenplay(shorter data) was reached.

```
helter-skelter, like a torrent that will and lathers, Ind's should be not be attell he personalion and play and demsinally
CONNER
omating eternal stop it. *

LEIGH *
The German,_ conquent relar alogation
(LEIggg)

CONNER
is a sign of the procuse of the world appears of self he had us. *

CONNER
The Pessimism is not political property and a comenning. *
belief in the forcow. In propers of the word to show by music who presents.

ERIN
130

helter-skelter, like a torrent that will this of _humanity_ and life, but whose oright and
On the wearthomed
What you don’t know’s dogmants and from the table the religious first second to paper who
ALLISON
animal _from the
Holy spirityed _absence_ for the everything?
animal fild happeners as the flacking intellectuar revolly mean hersto not it is the menss from be same into _trence there socies._
Thank you...
resplitering the time who would leave, who a (beat). The digned to be in its breath without and improveded by the _greaty_ in order of the presence of the fortions and weakness, and who are from the same
Erin’s apartment _trends into
ALLISON
conformination of us miditare wanterm scring characters. The fact a looks or an ablention, and its belong the lowior of courage what _strong "heart_ for the distour pich people to despirrity on same the oppresshing shakes us, there sellishness in the presence of the talking
problem natures the its the obgeratua it bitwence is
BRAD
who wearrabbilite who are over--a sensurare her posithing in a force 

```

### Stacked text file
In this step, I simply put the same length of hallmark movie script text and Nietzsche’s text together (103613 characters of each).

# Sources

Nietzsche, Friedrich. The Birth of Tragedy. Oxford University Press, 2008.

Garman Rick, and Jennifer Notas. The Bad Date Chronicles: The Complete Screenplay. Bad Date Productions, 2016.

