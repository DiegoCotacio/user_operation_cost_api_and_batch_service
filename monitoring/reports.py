


def get_column_mapping(**kwargs) -> ColumnMapping:

    column_mapping = ColumnMapping()
    column_mapping.target = kwargs['target_col']
    column_mapping.prediction = kwargs['prediction_col']
    column_mapping.numerical_features = kwargs['num_features']
    column_mapping.categorical_features = kwargs['cat_features']

    return column_mapping

