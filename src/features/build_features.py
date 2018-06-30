
## Modules for feature cleaning Titanic dataset
import pandas as pd

def simplify_sex(df):
    """Simplify sex and one hot encode them"""

    sex = pd.get_dummies(df.Sex, prefix = 'Sex')
    df = pd.concat([df, sex], axis = 1)
    return df

def simplify_class(df):
    """Simplify passenger class and one hot encode them"""

    new_class = pd.get_dummies(df.Pclass, prefix = 'Class')
    df = pd.concat([df, new_class], axis = 1)
    return df

def simplify_embarked(df):
    """Simplify where passengers embared from and one hot encode them"""

    embarked = pd.get_dummies(df.Embarked , prefix='Embarked' )
    df = pd.concat([df, embarked], axis=1)
    return df

def simplify_ages(df):
    """Seperate ages into bins and one hot encode them"""

    bins = (-1, 0,5,12,18,25,35,60,120)
    df.Age = df.Age.fillna(-0.5)
    group_names = ['Unknown', 'Baby', 'Child',
                   'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
    df.Age = pd.cut(df.Age, bins, labels = group_names)
    ages = pd.get_dummies(df.Age, prefix = 'Age')
    df = pd.concat([df,ages], axis = 1)
    return df

def simplify_fares(df):
    """Simplify fares and one hot encode them"""

    df.Fare = df.Fare.fillna(-0.5)
    bins = (-1, 0, 8, 15, 31, 1000)
    group_names = ['Unknown', '1_quartile', '2_quartile', '3_quartile', '4_quartile']
    df.Fare = pd.cut(df.Fare, bins, labels=group_names)
    fare = pd.get_dummies(df.Fare, prefix = 'Fare')
    df = pd.concat([df, fare], axis = 1)
    return df

def simpFamSize(df):
    bins = (0, 2, 4, 8, 20)
    group_names = ['Small', 'Medium', 'Large', 'ExtLarge']
    df.SibSp = pd.cut(df.SibSp, bins, labels = group_names)
    sibsp = pd.get_dummies(df.SibSp, prefix = 'Sibs')
    df = pd.concat([df, sibsp], axis = 1)
    return df

def drop_features(df):
    """Drop extra features not engineered"""

    return df.drop(['Ticket', 'Name', 'Age', 'Embarked', 'Pclass', 'Sex', 'Cabin', 'Fare', 'SibSp'], axis=1)

def produce_columns(df):
    """Reports columns for use in model."""
    columnsNames = []
    for i in df.columns:
        if i == 'Survived' or i == 'PassengerId':
            pass
        else:
            columnsNames.append(i)
    return columnsNames

def execute_cleaning(df):
    """One hot encode age, embarked, fares, class, sex and drop extra features."""

    df = simplify_ages(df)
    df = simplify_embarked(df)
    df = simplify_fares(df)
    df = simplify_class(df)
    df = simplify_sex(df)
    df = simpFamSize(df)
    df = drop_features(df)
    return df