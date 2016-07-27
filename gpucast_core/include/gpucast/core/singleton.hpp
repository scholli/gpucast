#ifndef GPUCAST_CORE_SINGLETON_HPP
#define GPUCAST_CORE_SINGLETON_HPP

namespace gpucast {

  template <typename T> class singleton {
  public:

    static T* instance() {
      static T instance;
      return &instance;
    };
  protected:

    singleton() {};
    virtual ~singleton() {};

  private:

    singleton(singleton const& copy) = delete;
    singleton& operator= (singleton const&) = delete;

  };

} // namespace gpucast

#endif  //GPUCAST_CORE_SINGLETON_HPP
