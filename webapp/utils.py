from webapp import db

def get_or_create(model, **filter_parameters):
    instance = model.query.filter_by(**filter_parameters).first()
    if not instance:
        instance = model(**filter_parameters)
        db.session.add(instance)
        db.session.commit()

    return instance
