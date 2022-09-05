from rhyme_finetuning import PoemsDataset

def test_save_load(tmp_path):
    data1 = PoemsDataset()
    data1.save(tmp_path / "data.pkl")
    data2 = PoemsDataset.load(tmp_path / "data.pkl")
    assert len(data1) == len(data2)
    assert all(s1 == s2 for s1, s2 in zip(data1, data2))