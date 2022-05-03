import pysmt
from pysmt.shortcuts import Symbol, And, Int, Ite, Real, LE, LT, Bool
from pysmt.typing import INT


def parse_dt_classifier(tree_model, interface, feature_names, quantize=True):
    df = tree_model.to_dataframe()
    df['node_idx'] = df.index
    assert df is not None
    input_symbols = {
        name: Symbol(name, INT) for name, dtype in zip(feature_names, interface[0])
    }
    
    def create_node_recur(row_dict):
        if row_dict['is_leaf']:
            return Bool(row_dict.get(True, 0) >= row_dict.get(False, 0))

        row_leq, row_gt = df[df['parent'] == row_dict['node_idx']].to_dict(orient='records')
        feat_symbol = row_dict['feature']
        if quantize:
            threshold_val = Int(int(row_dict['threshold']))
        else:
            threshold_val = Real(int(row_dict['threshold']))

        return Ite(
            LE(feat_symbol, threshold_val),
            create_node_recur(row_leq),
            create_node_recur(row_gt)
        )

    root_dict = df.to_dict(orient='records')[0]
    return create_node_recur(root_dict)
