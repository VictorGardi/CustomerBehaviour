class SpaceSwitcher():

    def __init__(self):
        pass

    def case_to_space(self, case, i):
        method_name = 'case_' + str(case)
        method = getattr(self, method_name)
        return method(i)

    def case_1(self, i):
        return i +1

    def case_2(self):
        pass

    def case_3(self):
        pass

s = SpaceSwitcher()

print(s.case_to_space(1, 1))
