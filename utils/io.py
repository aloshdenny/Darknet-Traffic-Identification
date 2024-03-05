import pickle


def save_model(model, path_save:str):
    if path_save is None:
        path_save = "model.pkl"
    else:
        if path_save[-4:] != ".pkl":
            path_save += ".pkl"
    pickle.dump(model, open(path_save, 'wb'))


def load_model(path_load:str):
    if path_load is None:
        path_load = "model.pkl"
    else:
        if path_load[-4:] != ".pkl":
            path_load += ".pkl"
    return pickle.load(open(path_load, 'rb'))
