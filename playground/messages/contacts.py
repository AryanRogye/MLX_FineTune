# type: ignore
"""
We have to ignore type checking for this file because the Contacts framework is not
available in the type stubs, and we are using it directly via PyObjC.
This file is only used on macOS, so it won't affect other platforms.
"""
import Contacts

def load_contacts() -> dict[str, str]:
    store = Contacts.CNContactStore.alloc().init()
    keys = [
        Contacts.CNContactGivenNameKey,
        Contacts.CNContactFamilyNameKey,
        Contacts.CNContactPhoneNumbersKey,
        Contacts.CNContactEmailAddressesKey,
    ]

    phone_map = {}  # cleaned number -> full name

    request = Contacts.CNContactFetchRequest.alloc().initWithKeysToFetch_(keys)

    def handler(contact, stop):
        name = f"{contact.givenName()} {contact.familyName()}".strip()
        for phone in contact.phoneNumbers():
            raw = phone.value().stringValue()
            clean = ''.join(filter(str.isdigit, raw))
            phone_map[clean] = name
            if len(clean) >= 10:
                phone_map[clean[-10:]] = name
            if len(clean) >= 7:
                phone_map[clean[-7:]] = name

    store.enumerateContactsWithFetchRequest_error_usingBlock_(request, None, handler)
    return phone_map
