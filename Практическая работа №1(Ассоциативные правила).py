from itertools import chain
import itertools
import pandas as pd

#columns = ['ТРУСЫ,БЮСТГАЛЬТЕР', 'НОСКИ', 'ЮБКА', 'БРЮКИ', 'БРАСЛЕТ,СЕРЕЖКИ', 'ФУТБОЛКА', 'ПИЖАМА', 'ТУФЛИ','БЛУЗКА', 'КРОССОВКИ']
columns = ['СМЕСЬ,БУТЫЛОЧКА', 'СОСКА', 'КОЛЯСКА', 'КРОВАТКА', 'САЛФЕТКИ,ПАМПЕРСЫ', 'АВТОКРЕСЛО', 'БОДИ', 'ПЕЛЕНКИ',
               'РАСПАШОНКИ', 'ГОРШОК']

def create_dict(): #создание словаря из файла
    d = {}
    with open("/Users/macbook/Desktop/6 семестр/Проектирование СППР/Программы/Shop(3).txt", encoding="utf8",
              errors='ignore') as f:
        for line in f:
            (key, val) = line.split()
            if int(key) not in d:
                d[int(key)] = [val]
            else:
                d[int(key)].append(val)
    return d

def f(d): #поиск уникальных значений товаров
    x = []
    for i in d.keys():
        x.extend(d[i])
    x = list(set(x))
    x.sort()
    return x

def sup(uniq, d):
    sup_d = {}
    for k in range(len(uniq)):
        for i in range(len(uniq[k]) - 1):
            for j in range(i + 1, len(uniq[k])):
                count = 0
                for key in d:
                    if uniq[k][i] in d[key] and uniq[k][j] in d[key]:
                        count += 1
                if (uniq[k][i], uniq[k][j]) not in sup_d:
                    sup_d[(uniq[k][i], uniq[k][j])] = count
                print('S({0} -> {1}) = {2}/{3} = {4}'.format(uniq[k][i], uniq[k][j], count, len(d), count/len(d)))
    return sup_d

def conf(uniq, d, sup_d):
    conf_d = {}
    for i in range(len(uniq) - 1):
        for j in range(i + 1, len(uniq)):
            count = 0
            for key in sup_d:
                if uniq[i] in key:
                    count += sup_d[key]
            if (uniq[i], uniq[j]) not in conf_d:
                conf_d[(uniq[i], uniq[j])] = count
            print('C({0} -> {1}) = {2}/{3} = {4}'.format(uniq[i], uniq[j], sup_d[(uniq[i], uniq[j])],
                                                         count, sup_d[(uniq[i], uniq[j])]/count))
    print(sup_d)

def create_norm_tranz(dict):
    '''Функция создания нормализованного образа множества транзакций'''
    mas_index = list(dict.keys())
    #df = pd.DataFrame(columns=columns).set_index(mas_index, inplace=True)
    df = pd.DataFrame(0, index=mas_index, columns=columns)
    #df = pd.DataFrame(columns=columns)
    for key in dict:
        for val in dict[key]:
            df.at[key, val] = 1
    df.loc['Сумма'] = df.sum()
    return df

def create_F2(df):
    '''Функция создания предметных наборов, состоящих из 2-х предметов'''
    d = {}
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            a = columns[i]
            b = columns[j]
            sum = len(df.loc[(df[a] == 1) & (df[b] == 1)])
            d['{0}; {1}'.format(a, b)] = sum
    two_subj_set = {key: value for key, value in d.items()
                 if value >= 4}
    new_d = {}
    for key in two_subj_set:
        buf = key.split('; ')
        buf.sort()
        new_d[tuple(buf)] = two_subj_set[key]
    f2 = list(new_d.keys())
    return f2, d

def create_F3(f2):
    f3 = list()
    for i in range(len(f2)):
        for j in range(i + 1, len(f2)):
            if f2[i][0] == f2[j][0]:
                f3.append([f2[i][0], f2[i][1], f2[j][1]])
    return f3

def func(df, f3, two_subj_set):
    all_comb = list()
    for mas in f3:
        for i in range(len(mas)):
            for j in range(i + 1, len(mas)):
                all_comb.append([mas[i], mas[j]])
    print(f3)
    for a in f3:
        for b in all_comb:
            conseq = list(set(a).difference(b))
            sum_all = len(df.loc[(df[b[0]] == 1) & (df[b[1]] == 1) & (df[conseq[0]] == 1)])
            count_conseq = len(df.loc[(df[b[0]] == 1) & (df[b[1]] == 1)])
            sup = sum_all/len(df[:-1])
            print('S({0} -> {1}) = {2}/{3} = {4}'.format(b, conseq, sum_all, len(df[:-1]), sup))
            sum = len(df.loc[(df[b[0]] == 1) & (df[b[1]] == 1) & (df[conseq[0]] == 1)])
            conf = sum_all/ count_conseq
            print('C({0} -> {1}) = {2}/{3} = {4}'.format(b, conseq, sum_all, count_conseq, conf))
    print('-------')
    for a in f3:
        for i in range(len(a)):
            for j in range(i + 1, len(a)):
                sum_all = len(df.loc[(df[a[i]] == 1) & (df[a[j]] == 1)])
                count_conseq = len(df.loc[(df[a[i]] == 1)])
                sup = sum_all / len(df[:-1])
                print('S({0} -> {1}) = {2}/{3} = {4}'.format(a[i], a[j], sum_all, len(df[:-1]), sup))
                conf = sum_all / count_conseq
                print('C({0} -> {1}) = {2}/{3} = {4}'.format(a[i], a[j], sum_all, count_conseq, conf))
                lift = conf / (df.at['Сумма', a[j]] / len(df[:-1]))
                print('L({0} -> {1}) = {2}/{3} = {4}'.format(a[i], a[j], conf,
                                                             (df.at['Сумма', a[j]] / len(df[:-1])), lift))
        for k in range(len(a) - 2, - 1, -1):
            sum_all = len(df.loc[(df[a[-1]] == 1) & (df[a[k]] == 1)])
            count_conseq = len(df.loc[(df[a[-1]] == 1)])
            sup = sum_all / len(df[:-1])
            print('S({0} -> {1}) = {2}/{3} = {4}'.format(a[-1], a[k], sum_all, len(df[:-1]), sup))
            conf = sum_all / count_conseq
            print('C({0} -> {1}) = {2}/{3} = {4}'.format(a[-1], a[k], sum_all, count_conseq, conf))
            lift = conf / (df.at['Сумма', a[k]] / len(df[:-1]))
            print('L({0} -> {1}) = {2}/{3} = {4}'.format(a[-1], a[k], conf,
                                                         (df.at['Сумма', a[k]] / len(df[:-1])), lift))



data = create_dict()
df = create_norm_tranz(data)
f2, two_subj_set = create_F2(df)
f3 = create_F3(f2)
func(df, f3, two_subj_set)


'''data = Associate(create_dict())
f(data.tranzactions)
sup_d = sup(f(data.tranzactions), data.tranzactions)
conf(f(data.tranzactions), data.tranzactions, sup_d)'''



