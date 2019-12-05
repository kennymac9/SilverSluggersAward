import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing

def make_prediction(position, league, year, model):

    train_data = pd.read_csv(league.lower() + '-' + position.lower() + '+2014.csv')
    if year == "2020":
        test_data = pd.read_csv('2020Predictions.csv')
    else:
        test_data = pd.read_csv(year + 'PredictionsAndActual.csv')

    years = [int(year) - 1, int(year) - 2, int(year) - 3]

    prev_year_data = train_data[train_data["Season"] == int(year) - 1]

    train_data = train_data[train_data["Season"].isin(years)]

    train_features = [ "R", "HR", "RBI", "AVG", "OPS"]
    if model == "linear":
        test_features = ["runsPredicted", "HRsPredicted", "RBIsPredicted", "AVGPredicted", "opsPredicted"]
    elif model == "linearSVR":
        test_features = ["linearSVRRuns", "linearSvrHR", "linearSvrRBI", "linearSvrAVG", "linearSvrOPS"]
    elif model == "SVR":
        test_features = ["svrRuns", "SvrHR", "svrRBI", "svrAVG", "svrOPS"]

    gnb = GaussianNB()

    gnb.fit(train_data[train_features].values, train_data["SS"])

    pred = gnb.predict_proba(test_data[test_features])

    players = {}

    max_prob = 0
    for i in range(len(pred)):
        if test_data.iloc[i, 0] in prev_year_data["Name"].values:
            players[test_data.iloc[i, 0]] = pred[i][1]

    players_sorted = sorted(players, key=players.get, reverse=True)

    return players_sorted

def recall(predictions, winners, year, value):
    counter = 0
    for league in winners:
        for position in winners[league]:
            if position == "ofs":
                for name in winners[league][position]['stats'][year]['names']:
                    try:
                        index_elem = predictions[league][position].index(name)
                        if index_elem <= value*3:
                            counter = counter + 1
                    except ValueError:
                        continue
            else:
                try:
                    index_elem = predictions[league][position].index(winners[league][position]['stats'][year]['name'])
                    if index_elem <= value:
                        counter = counter + 1
                except ValueError:
                    continue
    return counter

def ranking(predictions, winners, year):
    accum = 0
    for league in winners:
        for position in winners[league]:
            if position == 'ofs':
                for name in winners[league][position]['stats'][year]['names']:
                    try:
                        index_elem = predictions[league][position].index(name)
                        accum = accum + 1 / ((index_elem + 1) / 3)
                    except ValueError:
                        continue
            else:
                try:
                    index_elem = predictions[league][position].index(winners[league][position]['stats'][year]['name'])
                    accum = accum + 1 / (index_elem + 1)
                except ValueError:
                    continue
    return accum

def print_winners(winners, model, year):

    players_c_al = make_prediction("C", "AL", year, model)
    players_c_nl = make_prediction("C", "NL", year, model)
    players_1b_al = make_prediction("1B", "AL", year, model)
    players_1b_nl = make_prediction("1B", "NL", year, model)
    players_2b_al = make_prediction("2B", "AL", year, model)
    players_2b_nl = make_prediction("2B", "NL", year, model)
    players_3b_al = make_prediction("3B", "AL", year, model)
    players_3b_nl = make_prediction("3B", "NL", year, model)
    players_ss_al = make_prediction("SS", "AL", year, model)
    players_ss_nl = make_prediction("SS", "NL", year, model)
    players_of_al = make_prediction("OF", "AL", year, model)
    players_of_nl = make_prediction("OF", "NL", year, model)
    players_dh_al = make_prediction("DH", "AL", year, model)

    predictions = {
        'al': {
            'c': players_c_al,
            '1b': players_1b_al,
            '2b': players_2b_al,
            '3b': players_3b_al,
            'ss': players_ss_al,
            'ofs': players_of_al,
            'dh': players_dh_al
        },
        'nl': {
            'c': players_c_nl,
            '1b': players_1b_nl,
            '2b': players_2b_nl,
            '3b': players_3b_nl,
            'ss': players_ss_nl,
            'ofs': players_of_nl
        }
    }

    print(year + " " + model + '-----------------------------------------')
    print('C')
    print('AL: ' + players_c_al[0])
    print('NL: ' + players_c_nl[0])
    print('1B')
    print('AL: ' + players_1b_al[0])
    print('NL: ' + players_1b_nl[0])
    print('2B')
    print('AL: ' + players_2b_al[0])
    print('NL: ' + players_2b_nl[0])
    print('3B')
    print('AL: ' + players_3b_al[0])
    print('NL: ' + players_3b_nl[0])
    print('SS')
    print('AL: ' + players_ss_al[0])
    print('NL: ' + players_ss_nl[0])
    print('OF')
    print('AL: ' + players_of_al[0] + ', ' + players_of_al[1] + ', ' + players_of_al[2])
    print('NL: ' + players_of_nl[0] + ', ' + players_of_nl[1] + ', ' + players_of_nl[2])
    print('DH')
    print('AL: ' + players_dh_al[0])

    if year != '2020':
        print('Recall')
        recall_1 = recall(predictions, winners, year, 1)
        recall_2 = recall(predictions, winners, year, 2)
        recall_3 = recall(predictions, winners, year, 3)
        recall_4 = recall(predictions, winners, year, 4)
        recall_5 = recall(predictions, winners, year, 5)
        print(recall_1)
        print(recall_2)
        print(recall_3)
        print(recall_4)
        print(recall_5)
        print('Ranking')
        rank = ranking(predictions, winners, year)
        print(rank)

winners = {
    'al': {
        '1b': {
            'file': 'al-1b+2014.csv',
            'stats': {
                '2014': { 'name': 'Jose Abreu'},
                '2015': { 'name': 'Miguel Cabrera'},
                '2016': { 'name': 'Miguel Cabrera'},
                '2017': { 'name': 'Eric Hosmer'},
                '2018': { 'name': 'Jose Abreu'},
                '2019': { 'name': 'Carlos Santana'}
            }
        },
        '2b': {
            'file': 'al-2b+2014.csv',
            'stats': {
                '2014': { 'name': 'Jose Altuve'},
                '2015': { 'name': 'Jose Altuve'},
                '2016': { 'name': 'Jose Altuve'},
                '2017': { 'name': 'Jose Altuve'},
                '2018': { 'name': 'Jose Altuve'},
                '2019': { 'name': 'DJ LeMahieu'}
            }
        },
        '3b': {
            'file': 'al-3b+2014.csv',
            'stats': {
                '2014': { 'name': 'Adrian Beltre'},
                '2015': { 'name': 'Josh Donaldson'},
                '2016': { 'name': 'Josh Donaldson'},
                '2017': { 'name': 'Jose Ramirez'},
                '2018': { 'name': 'Jose Ramirez'},
                '2019': { 'name': 'Alex Bregman'}
            }
        },
        'ss': {
            'file': 'al-ss+2014.csv',
            'stats': {
                '2014': { 'name': 'Alexei Ramirez'},
                '2015': { 'name': 'Xander Bogaerts'},
                '2016': { 'name': 'Xander Bogaerts'},
                '2017': { 'name': 'Francisco Lindor'},
                '2018': { 'name': 'Francisco Lindor'},
                '2019': { 'name': 'Xander Bogaerts'}

            }
        },
        'ofs': {
            'file': 'al-of+2014.csv',
            'stats': {
                '2014': {
                    'names': ['Jose Bautista', 'Mike Trout', 'Michael Brantley']

                },
                '2015': {
                    'names': ['Nelson Cruz', 'Mike Trout', 'J.D. Martinez']

                },
                '2016': {
                    'names': ['Mookie Betts', 'Mike Trout', 'Mark Trumbo']

                },
                '2017': {
                    'names': ['Aaron Judge', 'George Springer', 'Justin Upton']

                },
                '2018': {
                    'names': ['Mookie Betts', 'Mike Trout', 'J.D. Martinez']

                },
                '2019': {
                    'names': ['Mookie Betts', 'Mike Trout', 'George Springer']

                }
            }
        },
        'c': {
            'file': 'al-c+2014.csv',
            'stats': {
                '2014': { 'name': 'Yan Gomes'},
                '2015': { 'name': 'Brian McCann'},
                '2016': { 'name': 'Salvador Perez'},
                '2017': { 'name': 'Gary Sanchez'},
                '2018': { 'name': 'Salvador Perez'},
                '2019': { 'name': 'Mitch Garver'}
            }
        },
        'dh': {
            'file': 'al-dh+2014.csv',
            'stats': {
                '2014': { 'name': 'Victor Martinez'},
                '2015': { 'name': 'Kendrys Morales'},
                '2016': { 'name': 'David Ortiz'},
                '2017': { 'name': 'Nelson Cruz'},
                '2018': { 'name': 'J.D. Martinez'},
                '2019': { 'name': 'Nelson Cruz'}
            }
        }
    },
    'nl': {
        '1b': {
            'file': 'nl-1b+2014.csv',
            'stats': {
                '2014': { 'name': 'Adrian Gonzalez'},
                '2015': { 'name': 'Paul Goldschmidt'},
                '2016': { 'name': 'Anthony Rizzo'},
                '2017': { 'name': 'Paul Goldschmidt'},
                '2018': { 'name': 'Paul Goldschmidt'},
                '2019': { 'name': 'Freddie Freeman'}
            }
        },
        '2b': {
            'file': 'nl-2b+2014.csv',
            'stats': {
                '2014': { 'name': 'Neil Walker'},
                '2015': { 'name': 'Dee Gordon'},
                '2016': { 'name': 'Daniel Murphy'},
                '2017': { 'name': 'Daniel Murphy'},
                '2018': { 'name': 'Javier Baez'},
                '2019': { 'name': 'Ozzie Albies'}
            }
        },
        '3b': {
            'file': 'nl-3b+2014.csv',
            'stats': {
                '2014': { 'name': 'Anthony Rendon'},
                '2015': { 'name': 'Nolan Arenado'},
                '2016': { 'name': 'Nolan Arenado'},
                '2017': { 'name': 'Nolan Arenado'},
                '2018': { 'name': 'Nolan Arenado'},
                '2019': { 'name': 'Anthony Rendon'}
            }
        },
        'ss': {
            'file': 'nl-ss+2014.csv',
            'stats': {
                '2014': { 'name': 'Ian Desmond'},
                '2015': { 'name': 'Brandom Crawford'},
                '2016': { 'name': 'Corey Seager'},
                '2017': { 'name': 'Corey Seager'},
                '2018': { 'name': 'Trevor Story'},
                '2019': { 'name': 'Trevor Story'}
            }
        },
        'ofs': {
            'file': 'nl-of+2014.csv',
            'stats': {
                '2014': {
                    'names': ['Andrew McCutchen', 'Giancarlo Stanton', 'Justin Upton']

                },
                '2015': {
                    'names': ['Andrew McCutchen', 'Bryce Harper', 'Carlos Gonzalez']

                },
                '2016': {
                    'names': ['Christian Yelich', 'Yoenis Cespedes', 'Charlie Blackmon']

                },
                '2017': {
                    'names': ['Marcell Ozuna', 'Giancarlo Stanton', 'Charlie Blackmon']

                },
                '2018': {
                    'names': ['Christian Yelich', 'Nick Markakis', 'David Peralta']

                },
                '2019': {
                    'names': ['Cody Bellinger', 'Christian Yelich', 'Ronald Acuna Jr.']

                }
            }
        },
        'c': {
            'file': 'nl-c+2014.csv',
            'stats': {
                '2014': { 'name': 'Buster Posey'},
                '2015': { 'name': 'Buster Posey'},
                '2016': { 'name': 'Wilson Ramos'},
                '2017': { 'name': 'Buster Posey'},
                '2018': { 'name': 'J.T. Realmuto'},
                '2019': { 'name': 'J.T. Realmuto'}
            }
        }
    }
}

print_winners(winners, "linear", "2017")
print_winners(winners, "linearSVR", "2017")
print_winners(winners, "SVR", "2017")

print_winners(winners, "linear", "2018")
print_winners(winners, "linearSVR", "2018")
print_winners(winners, "SVR", "2018")

print_winners(winners, "linear", "2019")
print_winners(winners, "linearSVR", "2019")
print_winners(winners, "SVR", "2019")

print_winners(winners, "linear", "2020")
print_winners(winners, "linearSVR", "2020")
print_winners(winners, "SVR", "2020")
