import xmlrpclib

s = xmlrpclib.ServerProxy('http://localhost:8000')
print s.is_even(2)