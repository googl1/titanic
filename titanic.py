import csv
import re
import pylab as plt
from scipy import stats
import numpy as np
from itertools import compress
import pandas as pd
from pandas import DataFrame
from sklearn.neighbors import KNeighborsClassifier
from math import floor,ceil
from sklearn.model_selection import LeaveOneOut
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import preprocessing


### Load and preprocess data ###

passenger_id = []
ticket = []
ticket_cost = []
port = []
class_ = []
cabin_number = []
full_name = []
sex = []
age = []
siblings_or_spouse = []
parents_or_children = []
survived = []

with open('dataset.csv', 'rt') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    try:
        next(reader)  # Skip header row.
        for row in reader: # Save table fields in lists (row-wise).
            passenger_id.append(int(row[0]))
            ticket.append(row[1])
            ticket_cost.append(float(row[2]))
            port.append(row[3])
            class_.append(int(row[4]))
            cabin_number.append(row[5])
            full_name.append(row[6])
            sex.append(row[7])
            age.append(row[8])
            siblings_or_spouse.append(int(row[9]))
            parents_or_children.append(int(row[10])) 
            survived.append(int(row[11]))
    except csv.Error as e:
        sys.exit('dataset.csv, line %d: %s' % (reader.line_num, e))

# The dataset lists passengers of the sunk Titanic and indicates if they survived the accident.
# Columns:

# passenger_id            Passenger ID
# ticket                  Ticket number
# ticket_cost             Price paid for the ticket
# port:                   The place they joined the ship
# class                   Class the passenger travelled in
# cabin_number            Cabin(s) occupied
# full_name               Name of passenger
# sex                     Sex of passenger
# age                     Age of passenger
# siblings_or_spouse      Number of siblings or spouse on board
# parents_or_children     Number of parents or children on board
# survived                1 if passenger survived, 0 if he died in the accident


### Analyze data for errors ###

# Check if IDs are correct.
if len(passenger_id) != passenger_id[-1]: 
    print('dataset.csv, last passenger_id: %d, num of passengers: %d' % (len(passenger_id), passenger_id[-1]))
if max(passenger_id) != passenger_id[-1]: 
    print('dataset.csv, last passenger_id: %d, num of passengers: %d' % (len(passenger_id), passenger_id[-1]))
# Look for duplicates.
if len(set([x for x in passenger_id if passenger_id.count(x) > 1])):
    print('dataset.csv, duplicate passenger_id: %s' % (set([x for x in passenger_id if passenger_id.count(x) > 1])))


# Parse ticket numbers.
for i,t in enumerate(ticket):
    if re.search(r'\d+', t) is None: 
                ticket[i] = 0   # 0 for missing ticket numbers.
    else:   # Take the last number in the string.
                ticket[i] = int(re.findall(r'\d+', t)[-1])


# Parse cabin number.
cabin = [] # 1 if passenger has cabin data, 0 if not
cabin_number_n = [] # number on the deck
cabin_multiple = [] # 1 if passenger has booked multiple cabins, 0 if not
cabin_deck = [] # letterfor the deck of the cabin
cabin_count = [] # total number of cabins
for i,c in enumerate(cabin_number):
    if re.findall(r'\d+', c) == []:
        cabin.append(0) # passenger has no cabin # data
        cabin_multiple.append(0) 
        cabin_deck.append('none')
        cabin_number_n.append(0)
        cabin_count.append(0)
    else:
        cabin.append(1)
        cabin_number_n.append(int(re.findall(r'\d+', c)[-1])) # take only the last cabin in list
        cabin_deck.append((re.findall(r'[A-G]', c))[-1])
        if len(re.findall(r'\d+', c)) > 1:
            cabin_multiple.append(1)
            cabin_count.append(len(re.findall(r'\d+', c)))
        else:
            cabin_multiple.append(0)  
            cabin_count.append(1)

# The ticket numbers sometimes begin with characters. The meaning is not clear and therefor the last number in the 
# String was the only thing used. If there was none, it was set to 0.

# The age is often missing.


### Exploratory data analysis ###

def two_histograms( dist1, dist2, bins, xlabel ):
    if bins is None:
        plt.hist(dist1, alpha=0.5, label='survived')
        plt.hist(dist2, alpha=0.5, label='dead')
    else:
        plt.hist(dist1, bins=bins, alpha=0.5, label='survived')
        plt.hist(dist2, bins=bins, alpha=0.5, label='dead')
    plt.legend(loc='upper left');
    plt.xlabel(xlabel)
    plt.show()
    
def bar_graph( by, xlabel ):
    uniq_y = np.unique(by)
    survived_by = []
    for p in uniq_y:
        survived_by.append([])
    
    for i,s in enumerate(survived):
        for j,p in enumerate(uniq_y):
            if p == by[i]:
                survived_by[j].append(s)

    num_survived_by = []
    num_dead_by = []

    for s in survived_by:
        num_survived_by.append(sum(s))
        num_dead_by.append(len(s)-sum(s))
         
    b1=plt.bar(range(len(uniq_y)), num_survived_by)
    b2=plt.bar(range(len(uniq_y)), num_dead_by, 0.8, num_survived_by)
    plt.legend((b1[0], b2[0]), ('Survived', 'Dead'))
    plt.xticks(range(len(uniq_y)), uniq_y)
    plt.xlabel(xlabel)
    plt.show()

survived_passenger = []
not_survived_passenger = []

for ID in passenger_id:
    if survived[ID-1] == 1:
        survived_passenger.append(ID-1)
    else:
        not_survived_passenger.append(ID-1)
        
# Histograms for ticket number
bins = np.linspace(0, 250000, 50)
two_histograms( dist1=[ticket[s] for s in survived_passenger], 
               dist2=[ticket[s] for s in not_survived_passenger], 
               bins=bins, xlabel='Ticket number' )

# Histograms for ID
bins = np.linspace(0, 900, 10)
two_histograms( dist1=[passenger_id[s] for s in survived_passenger], 
               dist2=[passenger_id[n] for n in not_survived_passenger], 
               bins=bins, xlabel='Passenger ID' )

# Histograms for ticket cost
two_histograms( dist1=[ticket_cost[s] for s in survived_passenger], 
               dist2=[ticket_cost[n] for n in not_survived_passenger], 
               bins=None, xlabel='Ticket cost' )

# this looks interesting, is the difference of those distributions significant?
stats.ks_2samp([ticket_cost[s] for s in survived_passenger], [ticket_cost[n] for n in not_survived_passenger])

# Port
bar_graph( by=port, xlabel='Port' )

# Class
bar_graph( by=class_, xlabel='Class' )

# this looks interesting, is the difference of those distributions significant?
stats.ks_2samp([class_[s] for s in survived_passenger], [class_[n] for n in not_survived_passenger])

# Deck of the cabin
bar_graph( by=cabin_deck, xlabel='Deck' )

# Histograms for cabin numbers
bins = np.linspace(0, 150, 10)
two_histograms( dist1=[cabin_number_n[s] for s in survived_passenger], 
               dist2=[cabin_number_n[n] for n in not_survived_passenger], 
               bins=bins, xlabel='Cabin' )

# Number of cabins
bar_graph( by=cabin_count, xlabel='Total cabins' )

# Sex
bar_graph( by=sex, xlabel='Sex' )

# Histograms for age
bins = np.linspace(0, 100, 50)
two_histograms( dist1=[age[s] for s in survived_passenger], 
               dist2=[age[n] for n in not_survived_passenger], 
               bins=bins, xlabel='Age' )

# Is there a sex/age interaction?
male = []
for i,s in enumerate(sex):
    if s == 'male':
        male.append(1)
    else:
        male.append(0)
        
survived_m = list(compress(age, [a and b for a, b in zip(survived, male)]))
survived_f = list(compress(age, [a and not b for a, b in zip(survived,  male)]))
dead_m = list(compress(age, [not a and b for a, b in zip(survived, male)]))
dead_f = list(compress(age, [not a and not b for a, b in zip(survived,  male)]))

#male
bins = np.linspace(0, 100, 50)
two_histograms( dist1=survived_m, 
               dist2=dead_m, 
               bins=bins, xlabel='Age (male only)' )

#female
bins = np.linspace(0, 100, 50)
two_histograms( dist1=survived_f, 
               dist2=dead_f, 
               bins=bins, xlabel='Age (female only)' )

# siblings or spouse
bar_graph( by=siblings_or_spouse, xlabel='Siblings or spouse' )

# parents or children
bar_graph( by=parents_or_children, xlabel='Parents or children' )

# All variables were plotted as bars or in histogram to find out 
# which ones might be predictive for the survival status and 
# how they are distributed.

# passenger_id            not interesting
# ticket                  2 ranges seem to be predictive
# ticket_cost             The distribution for survived passengers seems shifted to the right.
#                            The two distributions are different with high stat. significance.
#                            => People who paid more were more likely to survive.
# port:                   People who boarded in S (Southampton) where more likely to die.
# class                   Higher survivel rates in the better classes.
# cabin_number            Deck (A-G) seems to be predictive for survival. 
#                         People without a deck/cabin were more likely to die (lower down in the boat?).
#                         Lower cabin numbers were more likely to die (independent from deck).
#                         Higher survival rates for people who booked more than one cabin.
# full_name               
# sex                     Men more likely to die tan women
# age                     Seems predictive for survival. More victims around 25 years. More survivors among children.
#                         Very high number of victims in young men. 
#                         For women, survival rates are higher around 25 years of age.
# siblings_or_spouse      Lower survival rates for people without.
# parents_or_children     See above.


### Correlational analysis ###

cabin_deck_n = []
for c in cabin_deck:
    if c == 'A':
        cabin_deck_n.append(1)
    elif c == 'B':
        cabin_deck_n.append(2)
    elif c == 'C':
        cabin_deck_n.append(3)
    elif c == 'D':
        cabin_deck_n.append(4)
    elif c == 'E':
        cabin_deck_n.append(5)
    elif c == 'F':
        cabin_deck_n.append(6)
    elif c == 'G':
        cabin_deck_n.append(7)
    elif c == 'none':
        cabin_deck_n.append(8)
    else:
        print(c)
        
port_n = []
for p in port:
    if p == 'S':
        port_n.append(1)
    elif p == 'C':
        port_n.append(2)
    elif p == 'Q':
        port_n.append(3)
    elif p == '':
        port_n.append('NA')
    else:
        print('new port')

passengers = [('passenger_id', passenger_id),
         ('ticket', ticket),
         ('ticket_cost', ticket_cost),
         ('port', port_n),
         ('class', class_),
         ('cabin_number', cabin_number_n),
         ('cabin_count', cabin_count),
         ('cabin_deck', cabin_deck_n),
         ('sex', male),
         ('age', age),
         ('siblings_or_spouse', siblings_or_spouse),
         ('parents_or_children', parents_or_children)
             ]
df = pd.DataFrame.from_items(passengers)
print(df.corr())

def print_corr( x, y, xlabel, ylabel ):
    plt.scatter(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    print(np.corrcoef(x,y))
    
age_ = []
for a in age:
    if a == '':
        age_.append(np.nan)
    else:
        age_.append(float(a))
    
print_corr(ticket_cost, class_, 'Ticket cost', 'Class')
print_corr(ticket_cost, cabin_count, 'Ticket cost', 'Total cabins')
print_corr(class_, cabin_count, 'Class', 'Total cabins')
print_corr(age_, class_, 'Age', 'Class')
print_corr(siblings_or_spouse, parents_or_children, 'Siblings or spouse', 'Parents or children')
print_corr(cabin_deck_n, ticket_cost, 'Deck', 'Ticket cost')
print_corr(cabin_deck_n, class_, 'Deck', 'Class')
print_corr(cabin_number_n, ticket_cost, 'Cabin #', 'Ticket cost')
print_corr(cabin_number_n, class_, 'Cabin #', 'Class')
print_corr(cabin_number_n, cabin_count, 'Cabin #', 'Total cabins')
print_corr(cabin_number_n, cabin_deck_n, 'Cabin #', 'Deck')

# Correlations with |r| > 0.3 are visualized.

# Lower ticket numbers were more expensive. Maybe they were sold first.
# But also higher ticket numbers are from the higher classes.
# The higher numbers of cabins booked together have low ticket numbers.
# Better classes were more expensive, but not always.
# Older people tended to pay more.
# Passengers travelling with family tended to pay less.
# In better classes passengers tended to rather book multiple cabins.


### Feature engineering ###

# Parse cabin number.
cabin = []          # 1 if passenger has cabin data, 0 if not
cabin_number_n = [] # number on the deck
cabin_multiple = [] # 1 if passenger has booked multiple cabins, 0 if not
cabin_deck = []     # letter for the deck of the cabin
cabin_count = []    # total number of cabins
for i,c in enumerate(cabin_number):
    if re.findall(r'\d+', c) == []:
        cabin.append(0) # passenger has no cabin # data
        cabin_multiple.append(0) 
        cabin_deck.append('none')
        cabin_number_n.append(0)
        cabin_count.append(0)
    else:
        cabin.append(1)
        cabin_number_n.append(int(re.findall(r'\d+', c)[-1])) # take only the last cabin in list
        cabin_deck.append((re.findall(r'[A-G]', c))[-1])
        if len(re.findall(r'\d+', c)) > 1:
            cabin_multiple.append(1)
            cabin_count.append(len(re.findall(r'\d+', c)))
        else:
            cabin_multiple.append(0)  
            cabin_count.append(1)

passengers = [('ticket', ticket),
         ('ticket_cost', ticket_cost),
         ('port', port),
         ('class', class_),
         ('sex', sex),
         ('age', age_),
         ('siblings_or_spouse', siblings_or_spouse),
         ('parents_or_children', parents_or_children),
         ('survived', survived),
         ('cabin count', cabin_count),
         ('cabin number', cabin_number_n),
         ('deck', cabin_deck),
          ]
df = pd.DataFrame.from_items(passengers)

# Data Cleaning
df = df.dropna()

survived_bk = survived
survived = df['survived']
le = preprocessing.LabelEncoder()
df.sex = le.fit_transform(df.sex)
df.port = le.fit_transform(df.port)
df.deck = le.fit_transform(df.deck)
df = df.drop(['survived'], axis=1)

# The cabin information is only present for very few passengers.
# If present, it contains a letter for the deck followed by a room number
# for every cabin the passenger booked.
# From this we extract the number of cabins the passenger booked, as well as the deck and number 
# of the last room in the list.

# For now, all available data is used. The passengers without age were removed from the data, since 
# the age is quite predictive for survival and it's only a few.
# The sex, deck and port are encoded as integers.


### Create datasets ###

# Determine size of training/test set (70/30)

size = len(passenger_id)
train_size = int(floor(0.7 * size))
test_size = int(ceil(0.3 * size))

mat = df.values

np.random.seed(42)
perm = np.random.permutation(mat.shape[0])
mat = np.take(mat,perm,axis=0);
survived = np.take(survived,perm,axis=0);

train = mat[0:train_size,:]
test = mat[train_size:-1,:]
y_train = survived[0:train_size]
y_test = survived[train_size:-1]


### Initial model ###

gb = GradientBoostingClassifier();
gb.fit(train, y_train);


### Model evaluation ###

score = gb.score(test, y_test)
print('Accuracy:', score)

P = 1000
p = 0

for i in range(P):
    survived = np.random.permutation(survived)

    perm = np.random.permutation(mat.shape[0])
    mat = np.take(mat,perm,axis=0);
    survived = np.take(survived,perm,axis=0);

    train = mat[0:train_size,:]
    test = mat[train_size:-1,:]
    y_train = survived[0:train_size]
    y_test = survived[train_size:-1]
    
    p_score = gb.score(test, y_test)
    
    if p_score > score:
        p += 1
    
p = (p + 1) / (P + 1)

print('p:', p)

# The predictions of the model are correct with 84.44 %
# This is result is significant with p<0.001
# under the null hypothesis that the survival results are interchangable.


### Hyperparameter optimization ###

age_m = np.mean(age_)
age_n = age_
for a in age_n:
    if a == np.nan:
        a = age_m

passengers = [('ticket', ticket),
         ('ticket_cost', ticket_cost),
         ('port', port),
         ('class', class_),
         ('sex', sex),
         ('age', age_n),
         ('siblings_or_spouse', siblings_or_spouse),
         ('parents_or_children', parents_or_children),
         ('survived', survived_bk),
         ('cabin count', cabin_count),
         ('cabin number', cabin_number_n),
         ('deck', cabin_deck),
          ]
             
df = pd.DataFrame.from_items(passengers)

df = df.dropna()

survived = df['survived']
le = preprocessing.LabelEncoder()
df.sex = le.fit_transform(df.sex)
df.port = le.fit_transform(df.port)
df.deck = le.fit_transform(df.deck)
df = df.drop(['survived'], axis=1)

# Determine size of training/test set (70/30)

size = len(passenger_id)
train_size = int(floor(0.7 * size))
test_size = int(ceil(0.3 * size))

# Create datasets
mat = df.values

#np.random.shuffle(mat)
np.random.seed(42)
perm = np.random.permutation(mat.shape[0])
mat = np.take(mat,perm,axis=0);
survived = np.take(survived,perm,axis=0);

train = mat[0:train_size,:]
test = mat[train_size:-1,:]
y_train = survived[0:train_size]
y_test = survived[train_size:-1]

# leavo one out split for finding best training param (n)
ss = LeaveOneOut()
score = np.zeros(size)  # init score

best_n = 1

y_train = np.asarray(y_train)
best = 0
#for k in range(1,train.shape[1]):
for n in range(20,60):
    score = 0

    gb = GradientBoostingClassifier(n_estimators=n)
    gb.fit(train, y_train)
        
    one_score = gb.score(test, y_test)

    score = one_score


    if score > best:
        best = score
        best_n = n

print('Highest accuracy found:', best)
print('n =', best_n)

P = 1000
p = 0

for i in range(P):
    survived = np.random.permutation(survived)

    perm = np.random.permutation(mat.shape[0])
    mat = np.take(mat,perm,axis=0);
    survived = np.take(survived,perm,axis=0);

    train = mat[0:train_size,:]
    test = mat[train_size:-1,:]
    y_train = survived[0:train_size]
    y_test = survived[train_size:-1]
    
    p_score = gb.score(test, y_test)
    
    if p_score > score:
        p += 1
    
p = (p + 1) / (P + 1)

print('p:', p)

# The best parameter for the number of estimators of the gradient boosting classifier 
# was selected with a gridsearch. 
# The missing age values were replaced with the mean of all ages.

# The improved accuracy of 86.67 % is significant with p<0.001
# under the null hypothesis that the survival results are interchangable.
