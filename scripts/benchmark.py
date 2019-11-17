from shrynk import PandasCompressor

pdc = PandasCompressor()
weights = (1, 0, 0)

acc, result = pdc.validate(*weights)

pdc = PandasCompressor()
weights = (5, 1, 1)

acc, result = pdc.validate(*weights)


from shrynk import JsonCompressor

pdc = JsonCompressor()
weights = (1, 0, 0)

acc, result = pdc.validate(*weights)

from shrynk import BytesCompressor

pdc = BytesCompressor()
weights = (0, 1, 1)

acc, result = pdc.validate(*weights)
